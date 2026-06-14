from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import cv2
import face_recognition
import numpy as np
import sqlite3
import os
import time
import threading
from datetime import datetime

app = Flask(__name__)

# db
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, nim TEXT, nama TEXT, lokal TEXT, angkatan TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY, nama TEXT, nim TEXT, lokal TEXT, waktu TEXT, status TEXT)''')
    conn.commit()
    conn.close()

init_db()

# global var
known_face_encodings = []
known_face_names = []
known_face_nims = []
latest_raw_frame = None 
current_frame_for_ai = None
detected_faces_data = [] 

# load dataset
def load_known_faces():
    global known_face_encodings, known_face_names, known_face_nims
    known_face_encodings.clear()
    known_face_names.clear()
    known_face_nims.clear()
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        
    print("Loading dataset...")
    for filename in os.listdir('dataset'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join('dataset', filename)
            try:
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                
                name_part = os.path.splitext(filename)[0]
                if '_' in name_part:
                    nama, nim = name_part.split('_', 1)
                else:
                    nama = name_part
                    nim = "Unknown"
                    
                known_face_names.append(nama)
                known_face_nims.append(nim)
                print(f"[OK] {nama} di-load")
            except Exception as e:
                print(f"[ERROR] Skip {filename}: {e}")

load_known_faces()

# cam
camera = None
camera_index = 0

def get_camera():
    global camera, camera_index
    if camera is None:
        camera = cv2.VideoCapture(camera_index)
    return camera

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global camera, camera_index
    new_index = request.json.get('cam_index', 0)
    if camera is not None:
        camera.release() 
    camera_index = int(new_index)
    camera = cv2.VideoCapture(camera_index)
    return jsonify({"status": "success", "camera_index": camera_index})

# system deteksi 
def ai_worker():
    global current_frame_for_ai, detected_faces_data
    last_logged = {}
    
    while True:
        if current_frame_for_ai is not None:
            frame_to_scan = current_frame_for_ai.copy()
            small_frame = cv2.resize(frame_to_scan, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            temp_faces = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                name = "Tidak Dikenal"
                nim = ""

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    nim = known_face_nims[first_match_index]
                    
                    current_time = time.time()
                    
                    if nim not in last_logged or (current_time - last_logged[nim] > 15):
                        last_logged[nim] = current_time
                        sekarang = datetime.now()
                        waktu_str = sekarang.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Logika terlambattt
                        # kalo leweawt jam 8.00 dihitungnyat terlambbat
                        status_absen = "Terlambat" if sekarang.hour >= 8 else "Tepat Waktu"
                        
                        try:
                            conn = sqlite3.connect('database.db')
                            c = conn.cursor()
                            # lokal user ke tabel
                            c.execute("SELECT lokal FROM users WHERE nim=?", (nim,))
                            result = c.fetchone()
                            lokal_user = result[0] if result else "Unknown"
                            
                            c.execute("INSERT INTO logs (nama, nim, lokal, waktu, status) VALUES (?, ?, ?, ?, ?)", 
                                      (name, nim, lokal_user, waktu_str, status_absen))
                            conn.commit()
                            conn.close()
                            print(f"[ABSEN] {name} - {lokal_user} - {status_absen}")
                        except Exception as e:
                            print("Gagal tulis DB:", e)

                temp_faces.append((top * 4, right * 4, bottom * 4, left * 4, name))
            detected_faces_data = temp_faces
        time.sleep(0.05)

ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

# cam feed
def generate_frames():
    global latest_raw_frame, current_frame_for_ai, detected_faces_data
    cam = get_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break
        
        latest_raw_frame = frame.copy()
        current_frame_for_ai = frame.copy()

        for (top, right, bottom, left, name) in detected_faces_data:
            color = (93, 106, 140) if name == "Tidak Dikenal" else (93, 114, 85)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# coree
def mesin_pakar_sanksi(total_hadir, total_telat):
    if total_hadir == 0:
        return "Tidak Ada Keterangan (Alpha)", "badge-danger"
    elif total_telat == 0 and total_hadir > 0:
        return "Sangat Disiplin ( Nilai A)", "badge-success"
    elif total_telat >= 5:
        return "SP 2 & Pemanggilan Dosen Wali", "badge-danger"
    elif total_telat >= 3:
        return "Surat Peringatan 1 (SP 1)", "badge-warning"
    elif total_telat > 0 and total_telat < 3:
        return "Teguran Lisan oleh Dosen Lokal", "badge-info"
    
    return "Status Aman", "badge-normal"

#route

@app.route('/')
def index():
    return render_template('index.html') # landingpage

@app.route('/absensi')
def absensi():
    return render_template('absensi.html') # absen

@app.route('/registrasi')
def registrasi():
    return render_template('registrasi.html') # regiust

@app.route('/register', methods=['POST'])
def register():
    global latest_raw_frame
    nim = request.form['nim']
    nama = request.form['nama']
    lokal = request.form['lokal']
    angkatan = request.form['angkatan']
    
    if latest_raw_frame is not None:
        img_name = f"dataset/{nama}_{nim}.jpg"
        cv2.imwrite(img_name, latest_raw_frame)
        print(f"[SYSTEM] Foto {nama} ke-save!")
        load_known_faces()
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (nim, nama, lokal, angkatan) VALUES (?, ?, ?, ?)", (nim, nama, lokal, angkatan))
    conn.commit()
    conn.close()
    
    return redirect(url_for('registrasi'))

@app.route('/admin')
def admin():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT * FROM logs ORDER BY id DESC")
    logs = c.fetchall()

    c.execute("SELECT nim, nama, lokal FROM users")
    all_users = c.fetchall()
    
    rekap_sistem_pakar = []
    for user in all_users:
        nim_user, nama_user, lokal_user = user
        
        c.execute("SELECT COUNT(*) FROM logs WHERE nim=?", (nim_user,))
        total_hadir = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM logs WHERE nim=? AND status='Terlambat'", (nim_user,))
        total_telat = c.fetchone()[0]
        
        kesimpulan, warna_css = mesin_pakar_sanksi(total_hadir, total_telat)
        
        rekap_sistem_pakar.append({
            "nim": nim_user,
            "nama": nama_user,
            "lokal": lokal_user,
            "hadir": total_hadir,
            "telat": total_telat,
            "kesimpulan": kesimpulan,
            "warna": warna_css
        })
        
    conn.close()
    return render_template('admin.html', logs=logs, pakar_results=rekap_sistem_pakar)

# notif 
@app.route('/api/latest_log')
def latest_log():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT id, nama, nim, waktu, status FROM logs ORDER BY id DESC LIMIT 1")
    log = c.fetchone()
    conn.close()
    
    if log:
        return jsonify({"id": log[0], "nama": log[1], "nim": log[2], "waktu": log[3], "status": log[4]})
    return jsonify({"id": 0})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
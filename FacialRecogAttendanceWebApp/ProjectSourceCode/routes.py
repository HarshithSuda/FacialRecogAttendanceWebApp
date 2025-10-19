import os
import logging
import csv
import io
from datetime import datetime, timedelta, date
from flask import render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory, Response
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from sqlalchemy import func, and_, or_, desc, case, text, distinct
from sqlalchemy.orm import aliased

from app import app, db
from models import User, Student, ClassSession, Attendance, ProcessedImage, FaceEncoding
from facial_recognition import add_student_with_face, process_class_photo, process_multiple_photos, face_recognition

# Add current date to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Ensure upload directories exist
def ensure_dirs():
    for folder in ['uploads', 'processed', 'students']:
        directory = os.path.join(app.static_folder, folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

# Set upload folder configuration
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(app.static_folder, 'processed')
app.config['STUDENTS_FOLDER'] = os.path.join(app.static_folder, 'students')

# Create directories if they don't exist
ensure_dirs()

# Helper functions
def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter(
            or_(User.username == username, User.email == email)
        ).first()
        
        if existing_user:
            flash('Username or email already exists', 'danger')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        # Make the first registered user an admin
        if User.query.count() == 0:
            user.is_admin = True
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

# Main application routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent class sessions
    recent_sessions = ClassSession.query.order_by(ClassSession.date.desc()).limit(5).all()
    
    # Get attendance statistics
    today = date.today()
    week_ago = today - timedelta(days=7)
    
    today_attendance = ClassSession.query.filter(ClassSession.date == today).all()
    week_attendance = ClassSession.query.filter(ClassSession.date >= week_ago).all()
    
    total_students = Student.query.count()
    total_sessions = ClassSession.query.count()
    
    # Calculate attendance percentages
    today_percentage = 0
    week_percentage = 0
    
    if today_attendance:
        present_today = sum(session.present_students for session in today_attendance)
        total_today = sum(session.total_students for session in today_attendance)
        today_percentage = (present_today / total_today * 100) if total_today > 0 else 0
    
    if week_attendance:
        present_week = sum(session.present_students for session in week_attendance)
        total_week = sum(session.total_students for session in week_attendance)
        week_percentage = (present_week / total_week * 100) if total_week > 0 else 0
    
    return render_template('dashboard.html', 
                          recent_sessions=recent_sessions,
                          total_students=total_students,
                          total_sessions=total_sessions,
                          today_percentage=today_percentage,
                          week_percentage=week_percentage)

@app.route('/take-attendance', methods=['GET', 'POST'])
@login_required
def take_attendance():
    if request.method == 'POST':
        # Check if the form contains the class session information
        session_name = request.form.get('session_name')
        session_date = request.form.get('session_date')
        session_course = request.form.get('session_course', '')
        
        # Validate session data
        if not session_name or not session_date:
            flash('Session name and date are required', 'danger')
            return redirect(url_for('take_attendance'))
        
        try:
            session_date = datetime.strptime(session_date, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format', 'danger')
            return redirect(url_for('take_attendance'))
        
        # Create a new class session
        class_session = ClassSession(
            name=session_name,
            date=session_date,
            course=session_course,
            creator_id=current_user.id
        )
        db.session.add(class_session)
        db.session.flush()  # Get the session ID without committing
        
        # Check if files were uploaded
        if 'files' not in request.files:
            flash('No files selected', 'danger')
            db.session.rollback()
            return redirect(url_for('take_attendance'))
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            flash('No files selected', 'danger')
            db.session.rollback()
            return redirect(url_for('take_attendance'))
        
        # Filter valid image files
        valid_files = [f for f in files if f and allowed_file(f.filename)]
        if not valid_files:
            flash('No valid image files selected', 'danger')
            db.session.rollback()
            return redirect(url_for('take_attendance'))
        
        # Process all valid files for attendance
        success, results = process_multiple_photos(valid_files, class_session.id)
        
        if success:
            db.session.commit()
            flash('Images processed successfully!', 'success')
            return redirect(url_for('attendance', session_id=class_session.id))
        else:
            db.session.rollback()
            flash('Error processing images', 'danger')
            return redirect(url_for('take_attendance'))
    
    return render_template('take_attendance.html')

@app.route('/students', methods=['GET', 'POST'])
@login_required
def students():
    if request.method == 'POST':
        # Check if it's an Excel upload
        excel_file = request.files.get('excel_file')
        if excel_file and excel_file.filename.endswith(('.xlsx', '.xls')):
            try:
                return import_students_from_excel(excel_file)
            except Exception as e:
                flash(f'Error importing from Excel: {str(e)}', 'danger')
                return redirect(url_for('students'))
        
        # Regular student addition
        roll_number = request.form.get('roll_number')
        name = request.form.get('name')
        email = request.form.get('email', '')
        course = request.form.get('course', '')
        student_photo = request.files.get('student_photo')
        
        # Validate student data
        if not roll_number or not name:
            flash('Roll number and name are required', 'danger')
            return redirect(url_for('students'))
        
        # Check if student already exists
        existing_student = Student.query.filter_by(roll_number=roll_number).first()
        if existing_student:
            flash('A student with this roll number already exists', 'danger')
            return redirect(url_for('students'))
        
        # Validate and process the photo
        if not student_photo or student_photo.filename == '':
            flash('Student photo is required', 'danger')
            return redirect(url_for('students'))
        
        if not allowed_file(student_photo.filename):
            flash('Invalid file format. Please upload a .jpg, .jpeg, or .png file', 'danger')
            return redirect(url_for('students'))
        
        # Create the student and add their face
        success, message = add_student_with_face(roll_number, name, student_photo)
        
        if success:
            # Update additional student information
            student = Student.query.filter_by(roll_number=roll_number).first()
            student.email = email
            student.course = course
            db.session.commit()
            
            flash(message, 'success')
        else:
            flash(message, 'danger')
        
        return redirect(url_for('students'))
    
    # Get all students for the listing
    all_students = Student.query.order_by(Student.roll_number).all()
    return render_template('students.html', students=all_students)

# Take attendance by uploading student photos
@app.route('/update-face', methods=['GET', 'POST'])
@login_required
def update_face():
    if request.method == 'POST':
        roll_number = request.form.get('roll_number')
        student_photo = request.files.get('student_photo')
        
        # Validate inputs
        if not roll_number:
            flash('Student roll number is required', 'danger')
            return redirect(url_for('update_face'))
        
        # Find the student by roll number
        student = Student.query.filter_by(roll_number=roll_number).first()
        if not student:
            flash(f'No student found with roll number: {roll_number}', 'danger')
            return redirect(url_for('update_face'))
        
        # Validate the uploaded photo
        if not student_photo or student_photo.filename == '':
            flash('Student photo is required', 'danger')
            return redirect(url_for('update_face'))
        
        if not allowed_file(student_photo.filename):
            flash('Invalid file format. Please upload a .jpg, .jpeg, or .png file', 'danger')
            return redirect(url_for('update_face'))
        
        # Process the image to detect faces
        try:
            # Load and process the student image
            image = face_recognition.load_image_file(student_photo)
            if image is None:
                flash("Failed to load the student image", 'danger')
                return redirect(url_for('update_face'))
                
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                flash("No face detected in the image", 'danger')
                return redirect(url_for('update_face'))
                
            # Use the first detected face
            face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
            
            # Save the image file
            students_folder = current_app.config['STUDENTS_FOLDER']
            filename = secure_filename(f"{roll_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            image_path = os.path.join(students_folder, filename)
            
            # Convert BGR to RGB (OpenCV uses BGR format)
            import cv2
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Save the image file
            result = cv2.imwrite(image_path, image_bgr)
            if not result:
                logger.error(f"Failed to save image to {image_path}")
                flash("Failed to save student image", 'danger')
                return redirect(url_for('update_face'))
            
            # Add new face encoding to database
            encoding_record = FaceEncoding(
                student_id=student.id,
                encoding_data=face_encoding.tobytes(),
                image_filename=filename
            )
            db.session.add(encoding_record)
            db.session.commit()
            
            flash(f"Successfully marked attendance for student {student.name} ({student.roll_number})", 'success')
            return redirect(url_for('students'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error marking attendance: {str(e)}")
            flash(f"Error marking attendance: {str(e)}", 'danger')
            return redirect(url_for('update_face'))
    
    # For GET request: show the form
    # Get all students for the dropdown
    all_students = Student.query.order_by(Student.roll_number).all()
    
    # Check if we're looking for a specific student
    selected_roll = request.args.get('roll')
    selected_student = None
    
    if selected_roll:
        selected_student = Student.query.filter_by(roll_number=selected_roll).first()
    
    return render_template('update_face.html', 
                          students=all_students,
                          selected_student=selected_student)

@app.route('/students/faces/<roll_number>', methods=['GET'])
@login_required
def get_student_faces(roll_number):
    """Get all face images for a specific student"""
    try:
        student = Student.query.filter_by(roll_number=roll_number).first()
        
        if not student:
            return jsonify({"error": "Student not found"}), 404
        
        # Get all face encodings for this student
        faces = FaceEncoding.query.filter_by(student_id=student.id).all()
        
        face_data = []
        for face in faces:
            if face.image_filename:
                face_data.append({
                    "id": face.id,
                    "image_url": url_for('static', filename=f'students/{face.image_filename}'),
                    "created_at": face.created_at.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return jsonify({"student": {
            "id": student.id,
            "name": student.name,
            "roll_number": student.roll_number,
            "faces": face_data
        }})
    
    except Exception as e:
        logger.error(f"Error retrieving student faces: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/students/face/<int:face_id>/delete', methods=['POST'])
@login_required
def delete_student_face(face_id):
    """Delete a specific face encoding"""
    try:
        face = FaceEncoding.query.get(face_id)
        
        if not face:
            return jsonify({"error": "Face encoding not found"}), 404
        
        # Get the associated student for redirect
        student = Student.query.get(face.student_id)
        
        # Delete the image file if it exists
        if face.image_filename:
            image_path = os.path.join(current_app.config['STUDENTS_FOLDER'], face.image_filename)
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Delete the encoding record
        db.session.delete(face)
        db.session.commit()
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True, "message": "Face encoding deleted successfully"})
        else:
            flash("Face encoding deleted successfully", 'success')
            return redirect(url_for('update_face', roll=student.roll_number if student else None))
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting face encoding: {str(e)}")
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "error": str(e)}), 500
        else:
            flash(f"Error deleting face encoding: {str(e)}", 'danger')
            return redirect(url_for('update_face'))

def import_students_from_excel(excel_file):
    """Import students from an Excel file with roll numbers, names, and photo paths"""
    import pandas as pd
    import requests
    from io import BytesIO
    from PIL import Image
    import os
    import tempfile
    
    # Create a temporary directory to store downloaded images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Check if the required columns exist
        required_columns = ['roll_number', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            flash(f"Missing columns in Excel file: {', '.join(missing_columns)}", 'danger')
            return redirect(url_for('students'))
        
        # Track import stats
        total_rows = len(df)
        successful_imports = 0
        failed_imports = 0
        
        for _, row in df.iterrows():
            try:
                roll_number = str(row['roll_number'])
                name = row['name']
                
                # Check if student already exists
                existing_student = Student.query.filter_by(roll_number=roll_number).first()
                if existing_student:
                    logger.warning(f"Student with roll number {roll_number} already exists, skipping")
                    failed_imports += 1
                    continue
                
                # Create a new student without face encoding if photo_path is not provided
                if 'photo_path' not in df.columns or pd.isna(row['photo_path']):
                    # Create student without face encoding
                    student = Student(roll_number=roll_number, name=name)
                    if 'email' in df.columns and not pd.isna(row['email']):
                        student.email = row['email']
                    if 'course' in df.columns and not pd.isna(row['course']):
                        student.course = row['course']
                    
                    db.session.add(student)
                    db.session.commit()
                    successful_imports += 1
                    continue
                
                # If we reach here, photo_path is provided
                photo_path = row['photo_path']
                
                # Process the photo
                # Check if it's a URL or local path
                if photo_path.startswith(('http://', 'https://')):
                    # It's a URL, download the image
                    try:
                        response = requests.get(photo_path)
                        if response.status_code != 200:
                            logger.error(f"Failed to download image from URL: {photo_path}")
                            failed_imports += 1
                            continue
                        
                        # Create a temporary file
                        temp_file = os.path.join(temp_dir, f"{roll_number}.jpg")
                        with open(temp_file, 'wb') as f:
                            f.write(response.content)
                        
                        # Create an image file wrapper
                        class ImageFile:
                            def __init__(self, path):
                                self.path = path
                                self.filename = os.path.basename(path)
                            
                            def read(self):
                                with open(self.path, 'rb') as f:
                                    return f.read()
                            
                            def save(self, destination):
                                import shutil
                                shutil.copy2(self.path, destination)
                        
                        # Create student with face encoding
                        image_file = ImageFile(temp_file)
                        success, message = add_student_with_face(roll_number, name, image_file)
                        
                        if success:
                            # Update additional student information
                            student = Student.query.filter_by(roll_number=roll_number).first()
                            if 'email' in df.columns and not pd.isna(row['email']):
                                student.email = row['email']
                            if 'course' in df.columns and not pd.isna(row['course']):
                                student.course = row['course']
                            
                            db.session.commit()
                            successful_imports += 1
                        else:
                            logger.error(f"Failed to add student with face: {message}")
                            failed_imports += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing URL photo: {str(e)}")
                        failed_imports += 1
                
                else:
                    # It's a local path, just use the file directly
                    if not os.path.exists(photo_path):
                        logger.error(f"Photo file not found: {photo_path}")
                        failed_imports += 1
                        continue
                    
                    # Create an image file wrapper
                    class ImageFile:
                        def __init__(self, path):
                            self.path = path
                            self.filename = os.path.basename(path)
                        
                        def read(self):
                            with open(self.path, 'rb') as f:
                                return f.read()
                        
                        def save(self, destination):
                            import shutil
                            shutil.copy2(self.path, destination)
                    
                    # Create student with face encoding
                    image_file = ImageFile(photo_path)
                    success, message = add_student_with_face(roll_number, name, image_file)
                    
                    if success:
                        # Update additional student information
                        student = Student.query.filter_by(roll_number=roll_number).first()
                        if 'email' in df.columns and not pd.isna(row['email']):
                            student.email = row['email']
                        if 'course' in df.columns and not pd.isna(row['course']):
                            student.course = row['course']
                        
                        db.session.commit()
                        successful_imports += 1
                    else:
                        logger.error(f"Failed to add student with face: {message}")
                        failed_imports += 1
            
            except Exception as e:
                logger.error(f"Error importing student: {str(e)}")
                failed_imports += 1
        
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Show import results
        if successful_imports > 0:
            flash(f"Successfully imported {successful_imports} out of {total_rows} students", 'success')
        else:
            flash(f"Failed to import any students", 'danger')
        
        if failed_imports > 0:
            flash(f"Failed to import {failed_imports} students. Check the logs for details.", 'warning')
        
        return redirect(url_for('students'))
    
    except Exception as e:
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.error(f"Error importing from Excel: {str(e)}")
        flash(f"Error importing from Excel: {str(e)}", 'danger')
        return redirect(url_for('students'))

@app.route('/attendance/<int:session_id>')
@login_required
def attendance(session_id):
    """View attendance for a specific class session"""
    # Get the class session
    session = ClassSession.query.get_or_404(session_id)
    
    # Get all processed images for this session
    processed_images = ProcessedImage.query.filter_by(class_session_id=session_id).all()
    
    # Get all students and their attendance status for this session
    students = Student.query.order_by(Student.roll_number).all()
    attendance_records = []
    
    for student in students:
        attendance = Attendance.query.filter_by(
            student_id=student.id,
            class_session_id=session_id
        ).first()
        
        attendance_records.append((student, attendance))
    
    return render_template('attendance.html', 
                          session=session,
                          processed_images=processed_images,
                          attendance_records=attendance_records)

@app.route('/attendance/mark', methods=['POST'])
@login_required
def mark_manual_attendance():
    """Handle manual attendance marking"""
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        session_id = request.form.get('session_id')
        status = request.form.get('status')
        
        if not student_id or not session_id or not status:
            return jsonify({'success': False, 'message': 'Missing required parameters'}), 400
        
        try:
            student_id = int(student_id)
            session_id = int(session_id)
            
            # Check if student and session exist
            student = Student.query.get(student_id)
            session = ClassSession.query.get(session_id)
            
            if not student or not session:
                return jsonify({'success': False, 'message': 'Student or session not found'}), 404
            
            # Get existing attendance record if any
            attendance = Attendance.query.filter_by(
                student_id=student_id,
                class_session_id=session_id
            ).first()
            
            if status == 'present':
                if attendance:
                    # Update existing record
                    attendance.status = 'present'
                    attendance.marked_at = datetime.now()
                else:
                    # Create new record
                    attendance = Attendance(
                        student_id=student_id,
                        class_session_id=session_id,
                        status='present',
                        marked_at=datetime.now()
                    )
                    db.session.add(attendance)
                    
                # Update session stats
                session.present_students += 1
                
            elif status == 'absent':
                if attendance:
                    # Remove attendance record
                    db.session.delete(attendance)
                    
                    # Update session stats
                    if session.present_students > 0:
                        session.present_students -= 1
            
            db.session.commit()
            
            return jsonify({
                'success': True, 
                'message': f'Student {student.name} marked as {status}',
                'student': {
                    'id': student.id,
                    'name': student.name,
                    'roll_number': student.roll_number
                },
                'status': status
            })
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error marking attendance: {str(e)}")
            return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
    
    return jsonify({'success': False, 'message': 'Invalid request method'}), 405

@app.route('/reports')
@login_required
def reports():
    """View attendance reports and statistics"""
    try:
        # Get date range parameters
        start_date_str = request.args.get('start_date', '')
        end_date_str = request.args.get('end_date', '')
        course_filter = request.args.get('course', '')
        
        # Default to last 30 days if no date range provided
        today = date.today()
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            except ValueError:
                start_date = today - timedelta(days=30)
                end_date = today
        else:
            start_date = today - timedelta(days=30)
            end_date = today
        
        # Build the base query with date range filter
        date_filter = and_(ClassSession.date >= start_date, ClassSession.date <= end_date)
        
        # Add course filter if provided
        if course_filter:
            class_sessions = ClassSession.query.filter(
                date_filter,
                ClassSession.course == course_filter
            ).order_by(ClassSession.date.desc()).all()
        else:
            class_sessions = ClassSession.query.filter(
                date_filter
            ).order_by(ClassSession.date.desc()).all()
        
        # Get list of all courses for the filter dropdown
        courses = db.session.query(ClassSession.course).distinct().filter(
            ClassSession.course.isnot(None),
            ClassSession.course != ''
        ).order_by(ClassSession.course).all()
        courses = [course[0] for course in courses]
        
        # Get student-wise attendance stats
        student_attendance = []
        
        # Define a subquery to get the latest attendance date for each student
        latest_attendance = db.session.query(
            Attendance.student_id,
            func.max(ClassSession.date).label('last_date')
        ).join(
            ClassSession, Attendance.class_session_id == ClassSession.id
        ).filter(
            date_filter
        ).group_by(
            Attendance.student_id
        ).subquery()
        
        # Query for student-wise attendance within the date range
        student_stats = db.session.query(
            Student.id,
            Student.roll_number,
            Student.name,
            Student.course,
            func.count(Attendance.id).label('present_count'),
            func.count(distinct(ClassSession.id)).label('total_sessions'),
            func.max(ClassSession.date).label('last_attendance')
        ).outerjoin(
            Attendance, Student.id == Attendance.student_id
        ).outerjoin(
            ClassSession, and_(
                Attendance.class_session_id == ClassSession.id,
                date_filter
            )
        ).group_by(
            Student.id
        ).all()
        
        # Process student stats to calculate percentages
        for student in student_stats:
            # Count total sessions in the date range
            if course_filter:
                total_sessions = ClassSession.query.filter(
                    date_filter,
                    ClassSession.course == course_filter
                ).count()
            else:
                total_sessions = ClassSession.query.filter(date_filter).count()
            
            # Add to the list with calculated fields
            student_attendance.append({
                'id': student.id,
                'roll_number': student.roll_number,
                'name': student.name,
                'course': student.course,
                'present_count': student.present_count,
                'total_sessions': total_sessions,
                'last_attendance': student.last_attendance
            })
        
        # Get course-wise attendance stats
        course_attendance = []
        
        # Get all courses with at least one student
        all_courses = db.session.query(Student.course).distinct().filter(
            Student.course.isnot(None),
            Student.course != ''
        ).all()
        all_courses = [course[0] for course in all_courses]
        
        # Add an entry for students without a course
        all_courses.append(None)
        
        for course_name in all_courses:
            display_name = course_name or 'Uncategorized'
            
            # Count students in this course
            student_count = Student.query.filter(
                Student.course == course_name if course_name else or_(
                    Student.course.is_(None),
                    Student.course == ''
                )
            ).count()
            
            # Get sessions for this course
            if course_name:
                sessions = ClassSession.query.filter(
                    date_filter,
                    ClassSession.course == course_name
                ).all()
            else:
                sessions = ClassSession.query.filter(
                    date_filter,
                    or_(
                        ClassSession.course.is_(None),
                        ClassSession.course == ''
                    )
                ).all()
            
            session_count = len(sessions)
            
            # Calculate average attendance
            if session_count > 0:
                total_present = sum(session.present_students for session in sessions)
                total_students = sum(session.total_students for session in sessions)
                avg_attendance = (total_present / total_students * 100) if total_students > 0 else 0
                last_session = max(sessions, key=lambda s: s.date).date if sessions else None
            else:
                avg_attendance = 0
                last_session = None
            
            course_attendance.append({
                'name': display_name,
                'student_count': student_count,
                'session_count': session_count,
                'avg_attendance': avg_attendance,
                'last_session': last_session
            })
        
        # Get students with consecutive absences (3+)
        consecutive_absences = []
        
        # Get all active students
        students = Student.query.all()
        
        for student in students:
            # Get all sessions in date range ordered by date
            sessions_in_range = ClassSession.query.filter(date_filter).order_by(ClassSession.date).all()
            
            # Skip if no sessions
            if not sessions_in_range:
                continue
            
            # Check attendance for each session
            absent_streak = 0
            max_absent_streak = 0
            last_present = None
            
            for session in sessions_in_range:
                attendance = Attendance.query.filter_by(
                    student_id=student.id,
                    class_session_id=session.id
                ).first()
                
                if attendance:
                    # Student was present
                    absent_streak = 0
                    last_present = session.date
                else:
                    # Student was absent
                    absent_streak += 1
                    max_absent_streak = max(max_absent_streak, absent_streak)
            
            # If student has 3+ consecutive absences, add to the list
            if max_absent_streak >= 3:
                consecutive_absences.append({
                    'id': student.id,
                    'roll_number': student.roll_number,
                    'name': student.name,
                    'course': student.course,
                    'consecutive_absences': max_absent_streak,
                    'last_attendance': last_present
                })
        
        # Get students with less than 80% attendance
        low_attendance = []
        
        for student in student_attendance:
            if student['total_sessions'] > 0:
                attendance_percentage = (student['present_count'] / student['total_sessions']) * 100
                if attendance_percentage < 80:
                    low_attendance.append(student)
        
        return render_template('reports.html',
                              sessions=class_sessions,
                              student_attendance=student_attendance,
                              course_attendance=course_attendance,
                              low_attendance=low_attendance,
                              consecutive_absences=consecutive_absences,
                              courses=courses,
                              start_date=start_date,
                              end_date=end_date,
                              selected_course=course_filter)
    
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")
        flash(f"Error generating reports: {str(e)}", 'danger')
        return redirect(url_for('dashboard'))

@app.route('/export/attendance/<string:report_type>')
@login_required
def export_attendance(report_type):
    """Export attendance data as CSV"""
    try:
        # Get date range parameters
        start_date_str = request.args.get('start_date', '')
        end_date_str = request.args.get('end_date', '')
        course_filter = request.args.get('course', '')
        
        # Default to last 30 days if no date range provided
        today = date.today()
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            except ValueError:
                start_date = today - timedelta(days=30)
                end_date = today
        else:
            start_date = today - timedelta(days=30)
            end_date = today
        
        # Build the date range filter
        date_filter = and_(ClassSession.date >= start_date, ClassSession.date <= end_date)
        
        # Create a CSV output file in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        if report_type == 'daily':
            # Export daily attendance
            writer.writerow(['Date', 'Session Name', 'Course', 'Students Present', 'Total Students', 'Attendance %'])
            
            if course_filter:
                sessions = ClassSession.query.filter(
                    date_filter,
                    ClassSession.course == course_filter
                ).order_by(ClassSession.date.desc()).all()
            else:
                sessions = ClassSession.query.filter(date_filter).order_by(ClassSession.date.desc()).all()
            
            for session in sessions:
                attendance_percentage = (session.present_students / session.total_students * 100) if session.total_students > 0 else 0
                writer.writerow([
                    session.date.strftime('%Y-%m-%d'),
                    session.name,
                    session.course or 'N/A',
                    session.present_students,
                    session.total_students,
                    f"{attendance_percentage:.1f}%"
                ])
            
            filename = f"daily_attendance_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        
        elif report_type == 'student':
            # Export student-wise attendance
            writer.writerow(['Roll Number', 'Name', 'Course', 'Present Days', 'Total Sessions', 'Attendance %', 'Last Attendance'])
            
            # Query for student-wise attendance stats
            student_stats = db.session.query(
                Student.id,
                Student.roll_number,
                Student.name,
                Student.course,
                func.count(Attendance.id).label('present_count'),
                func.max(ClassSession.date).label('last_attendance')
            ).outerjoin(
                Attendance, Student.id == Attendance.student_id
            ).outerjoin(
                ClassSession, and_(
                    Attendance.class_session_id == ClassSession.id,
                    date_filter
                )
            ).group_by(
                Student.id
            ).all()
            
            # Count total sessions in date range
            if course_filter:
                total_sessions = ClassSession.query.filter(
                    date_filter,
                    ClassSession.course == course_filter
                ).count()
            else:
                total_sessions = ClassSession.query.filter(date_filter).count()
            
            for student in student_stats:
                attendance_percentage = (student.present_count / total_sessions * 100) if total_sessions > 0 else 0
                writer.writerow([
                    student.roll_number,
                    student.name,
                    student.course or 'N/A',
                    student.present_count,
                    total_sessions,
                    f"{attendance_percentage:.1f}%",
                    student.last_attendance.strftime('%Y-%m-%d') if student.last_attendance else 'Never'
                ])
            
            filename = f"student_attendance_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        
        elif report_type == 'course':
            # Export course-wise attendance
            writer.writerow(['Course', 'Total Students', 'Total Sessions', 'Average Attendance %', 'Last Session'])
            
            # Get all courses with at least one student
            all_courses = db.session.query(Student.course).distinct().filter(
                Student.course.isnot(None),
                Student.course != ''
            ).all()
            all_courses = [course[0] for course in all_courses]
            
            # Add an entry for students without a course
            all_courses.append(None)
            
            for course_name in all_courses:
                display_name = course_name or 'Uncategorized'
                
                # Count students in this course
                student_count = Student.query.filter(
                    Student.course == course_name if course_name else or_(
                        Student.course.is_(None),
                        Student.course == ''
                    )
                ).count()
                
                # Get sessions for this course
                if course_name:
                    sessions = ClassSession.query.filter(
                        date_filter,
                        ClassSession.course == course_name
                    ).all()
                else:
                    sessions = ClassSession.query.filter(
                        date_filter,
                        or_(
                            ClassSession.course.is_(None),
                            ClassSession.course == ''
                        )
                    ).all()
                
                session_count = len(sessions)
                
                # Calculate average attendance
                if session_count > 0:
                    total_present = sum(session.present_students for session in sessions)
                    total_students = sum(session.total_students for session in sessions)
                    avg_attendance = (total_present / total_students * 100) if total_students > 0 else 0
                    last_session = max(sessions, key=lambda s: s.date).date if sessions else None
                else:
                    avg_attendance = 0
                    last_session = None
                
                writer.writerow([
                    display_name,
                    student_count,
                    session_count,
                    f"{avg_attendance:.1f}%",
                    last_session.strftime('%Y-%m-%d') if last_session else 'N/A'
                ])
            
            filename = f"course_attendance_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        
        elif report_type == 'alerts':
            # Export attendance alerts
            writer.writerow(['Roll Number', 'Name', 'Course', 'Present Days', 'Total Sessions', 'Attendance %', 'Consecutive Absences', 'Last Attendance'])
            
            # First get students with less than 80% attendance
            student_stats = db.session.query(
                Student.id,
                Student.roll_number,
                Student.name,
                Student.course,
                func.count(Attendance.id).label('present_count'),
                func.max(ClassSession.date).label('last_attendance')
            ).outerjoin(
                Attendance, Student.id == Attendance.student_id
            ).outerjoin(
                ClassSession, and_(
                    Attendance.class_session_id == ClassSession.id,
                    date_filter
                )
            ).group_by(
                Student.id
            ).all()
            
            # Count total sessions in date range
            if course_filter:
                total_sessions = ClassSession.query.filter(
                    date_filter,
                    ClassSession.course == course_filter
                ).count()
            else:
                total_sessions = ClassSession.query.filter(date_filter).count()
            
            # Calculate consecutive absences for each student
            for student in student_stats:
                attendance_percentage = (student.present_count / total_sessions * 100) if total_sessions > 0 else 0
                
                if attendance_percentage < 80 or total_sessions - student.present_count >= 3:
                    # Calculate consecutive absences
                    consecutive_absences = 0
                    sessions_in_range = ClassSession.query.filter(date_filter).order_by(ClassSession.date).all()
                    
                    absent_streak = 0
                    max_absent_streak = 0
                    
                    for session in sessions_in_range:
                        attendance = Attendance.query.filter_by(
                            student_id=student.id,
                            class_session_id=session.id
                        ).first()
                        
                        if attendance:
                            absent_streak = 0
                        else:
                            absent_streak += 1
                            max_absent_streak = max(max_absent_streak, absent_streak)
                    
                    consecutive_absences = max_absent_streak
                    
                    writer.writerow([
                        student.roll_number,
                        student.name,
                        student.course or 'N/A',
                        student.present_count,
                        total_sessions,
                        f"{attendance_percentage:.1f}%",
                        consecutive_absences,
                        student.last_attendance.strftime('%Y-%m-%d') if student.last_attendance else 'Never'
                    ])
            
            filename = f"attendance_alerts_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        
        else:
            return jsonify({'error': 'Invalid report type'}), 400
        
        # Seek to the beginning of the in-memory file
        output.seek(0)
        
        # Create response
        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
    
    except Exception as e:
        logger.error(f"Error exporting attendance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/student/attendance/<int:student_id>')
@login_required
def student_attendance_detail(student_id):
    """Get detailed attendance for a specific student"""
    try:
        # Get student
        student = Student.query.get_or_404(student_id)
        
        # Get date range parameters
        start_date_str = request.args.get('start_date', '')
        end_date_str = request.args.get('end_date', '')
        
        # Default to last 30 days if no date range provided
        today = date.today()
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            except ValueError:
                start_date = today - timedelta(days=30)
                end_date = today
        else:
            start_date = today - timedelta(days=30)
            end_date = today
        
        # Get all sessions in date range
        sessions = ClassSession.query.filter(
            ClassSession.date >= start_date,
            ClassSession.date <= end_date
        ).order_by(ClassSession.date).all()
        
        # Check attendance for each session
        attendance_data = []
        
        for session in sessions:
            attendance = Attendance.query.filter_by(
                student_id=student.id,
                class_session_id=session.id
            ).first()
            
            attendance_data.append({
                'session_id': session.id,
                'session_name': session.name,
                'date': session.date.strftime('%Y-%m-%d'),
                'course': session.course or 'N/A',
                'status': 'Present' if attendance else 'Absent',
                'marked_at': attendance.marked_at.strftime('%Y-%m-%d %H:%M:%S') if attendance else None
            })
        
        # Calculate overall attendance percentage
        total_sessions = len(sessions)
        present_count = sum(1 for item in attendance_data if item['status'] == 'Present')
        attendance_percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0
        
        # Find consecutive absences
        absent_streak = 0
        max_absent_streak = 0
        
        for item in attendance_data:
            if item['status'] == 'Absent':
                absent_streak += 1
                max_absent_streak = max(max_absent_streak, absent_streak)
            else:
                absent_streak = 0
        
        return jsonify({
            'student': {
                'id': student.id,
                'roll_number': student.roll_number,
                'name': student.name,
                'course': student.course or 'N/A'
            },
            'attendance_summary': {
                'total_sessions': total_sessions,
                'present_count': present_count,
                'absent_count': total_sessions - present_count,
                'attendance_percentage': round(attendance_percentage, 1),
                'consecutive_absences': max_absent_streak
            },
            'date_range': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            },
            'attendance_data': attendance_data
        })
    
    except Exception as e:
        logger.error(f"Error getting student attendance detail: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Static file serving routes
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)
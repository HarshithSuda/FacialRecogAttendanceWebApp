import os
import cv2
import numpy as np
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app
from models import Student, FaceEncoding, Attendance, ClassSession, ProcessedImage
from app import db

# Temporary face recognition mockup since we can't install the actual face_recognition library
# This is a simplified version for demonstration purposes
class FaceRecognitionMock:
    @staticmethod
    def load_image_file(image_path):
        """Load an image file (.jpg, .png, etc) into a numpy array"""
        try:
            # Check if image_path is a file-like object (from form upload)
            if hasattr(image_path, 'read'):
                # Save to a temporary file first
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
                    temp_name = temp.name
                    image_path.save(temp_name)
                # Read the image from the temporary file
                image = cv2.imread(temp_name)
                if image is None:
                    raise ValueError(f"Failed to load image from temporary file: {temp_name}")
                return image
            
            # Check if image_path is a string path
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image from path: {image_path}")
                return image
            
            # If it's neither a file object nor a string
            raise ValueError(f"Unsupported image input type: {type(image_path)}")
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            raise
    
    @staticmethod
    def face_locations(image):
        """Returns an array of bounding boxes of human faces in an image"""
        # Using OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert from (x, y, w, h) to (top, right, bottom, left)
        locations = []
        for (x, y, w, h) in faces:
            locations.append((y, x+w, y+h, x))
        
        return locations
    
    @staticmethod
    def face_encodings(image, known_face_locations=None):
        """Returns the 128-dimension face encoding for each face in the image"""
        # In a real implementation, this would use a face recognition model
        # Here we'll just return a random encoding for each face
        if known_face_locations is None:
            known_face_locations = FaceRecognitionMock.face_locations(image)
        
        encodings = []
        for face_location in known_face_locations:
            # Generate a deterministic "encoding" based on the face location
            # This is just a placeholder and won't actually recognize faces properly
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            # Resize to standard size and flatten to create a simple "encoding"
            resized = cv2.resize(face_image, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            encoding = gray.flatten() / 255.0  # Normalize to 0-1
            encodings.append(encoding)
        
        return encodings
    
    @staticmethod
    def compare_faces(known_face_encodings, face_encoding, tolerance=0.6):
        """Compare a list of face encodings against a candidate encoding"""
        # In a real implementation, this would calculate distances between face embeddings
        # Here we'll simulate matches with some randomness but also some consistency
        if not known_face_encodings:
            return []
        
        matches = []
        for known_encoding in known_face_encodings:
            # Calculate a simplified "distance" - in reality this would be a proper distance metric
            if len(known_encoding) == len(face_encoding):
                # Use mean squared error as a simple distance metric
                distance = np.mean((known_encoding - face_encoding) ** 2)
                match = distance < tolerance
            else:
                match = False
            matches.append(match)
        
        return matches
    
    @staticmethod
    def face_distance(known_face_encodings, face_encoding):
        """Calculate euclidean distance between faces"""
        if not known_face_encodings:
            return np.array([])
        
        distances = []
        for known_encoding in known_face_encodings:
            if len(known_encoding) == len(face_encoding):
                # Use mean squared error as a simple distance metric
                distance = np.mean((known_encoding - face_encoding) ** 2)
            else:
                distance = 1.0  # Maximum distance if shapes don't match
            distances.append(distance)
        
        return np.array(distances)

# Replace the face_recognition import with our mock
face_recognition = FaceRecognitionMock()

logger = logging.getLogger(__name__)

def load_known_faces():
    """Load all known faces from the database"""
    known_face_encodings = []
    known_student_ids = []
    known_face_roll_numbers = []  # Added to store roll numbers
    
    # Get all students with face encodings
    students_with_faces = db.session.query(Student, FaceEncoding).join(
        FaceEncoding, Student.id == FaceEncoding.student_id
    ).all()
    
    for student, face_encoding in students_with_faces:
        known_face_encodings.append(np.frombuffer(face_encoding.encoding_data))
        known_student_ids.append(student.id)
        known_face_roll_numbers.append(student.roll_number)
    
    return known_face_encodings, known_student_ids, known_face_roll_numbers

def add_student_with_face(roll_number, name, image_file):
    """Add a new student with their face encoding to the database"""
    try:
        # Ensure the directory exists
        students_folder = current_app.config['STUDENTS_FOLDER']
        os.makedirs(students_folder, exist_ok=True)
        
        # Load and process the student image
        image = face_recognition.load_image_file(image_file)
        if image is None:
            return False, "Failed to load the student image"
            
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return False, "No face detected in the image"
            
        # Use the first detected face
        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        
        # Check if student exists, if not create one
        student = Student.query.filter_by(roll_number=roll_number).first()
        if not student:
            student = Student(roll_number=roll_number, name=name)
            db.session.add(student)
            db.session.flush()  # Get the student ID without committing
        
        # Save the image file
        filename = secure_filename(f"{roll_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        image_path = os.path.join(students_folder, filename)
        
        # Convert BGR to RGB (OpenCV uses BGR format)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the image file
        result = cv2.imwrite(image_path, image_bgr)
        if not result:
            logger.error(f"Failed to save image to {image_path}")
            return False, f"Failed to save student image"
        
        # Add face encoding to database
        encoding_record = FaceEncoding(
            student_id=student.id,
            encoding_data=face_encoding.tobytes(),
            image_filename=filename
        )
        db.session.add(encoding_record)
        db.session.commit()
        
        return True, f"Successfully added face for student {name} ({roll_number})"
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding student face: {str(e)}")
        return False, f"Error adding student: {str(e)}"

def process_class_photo(image_file, class_session_id):
    """Process a class photo to mark attendance for a specific session"""
    try:
        # Get the class session
        class_session = ClassSession.query.get(class_session_id)
        if not class_session:
            return False, "Class session not found"
        
        # Load known faces
        known_face_encodings, known_student_ids, known_face_roll_numbers = load_known_faces()
        if not known_face_encodings:
            return False, "No registered student faces found"
        
        # Ensure directories exist
        uploads_folder = current_app.config['UPLOAD_FOLDER']
        processed_folder = current_app.config['PROCESSED_FOLDER']
        os.makedirs(uploads_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
        
        # Load and process the class image
        original_filename = secure_filename(image_file.filename)
        image_path = os.path.join(uploads_folder, original_filename)
        
        # Save the uploaded file
        try:
            image_file.save(image_path)
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            return False, f"Failed to save uploaded file: {str(e)}"
        
        # Load image for face recognition
        try:
            image = face_recognition.load_image_file(image_path)
            if image is None:
                return False, "Failed to load the class photo"
        except Exception as e:
            logger.error(f"Error loading image for face recognition: {str(e)}")
            return False, f"Failed to process image: {str(e)}"
            
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return False, "No faces detected in the class photo"
            
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # For visualization, convert to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Track recognized students
        recognized_students = set()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            # Find the best match
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    student_id = known_student_ids[best_match_index]
                    roll_number = known_face_roll_numbers[best_match_index]
                    student = Student.query.get(student_id)
                    
                    if student:
                        recognized_students.add(student_id)
                        
                        # Mark attendance for this student
                        attendance = Attendance.query.filter_by(
                            student_id=student_id,
                            class_session_id=class_session_id
                        ).first()
                        
                        if not attendance:
                            attendance = Attendance(
                                student_id=student_id,
                                class_session_id=class_session_id,
                                status='present'
                            )
                            db.session.add(attendance)
                        
                        # Draw rectangle and student info on the image
                        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image_bgr, roll_number, (left + 6, bottom - 6), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                # Draw rectangle for unrecognized faces
                else:
                    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Save the processed image
        processed_filename = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}_{original_filename}"
        processed_path = os.path.join(processed_folder, processed_filename)
        
        # Save the processed image
        result = cv2.imwrite(processed_path, image_bgr)
        if not result:
            logger.error(f"Failed to save processed image to {processed_path}")
            return False, "Failed to save processed image"
        
        # Update class session stats
        total_students = Student.query.count()
        present_students = len(recognized_students)
        class_session.total_students = total_students
        class_session.present_students = present_students
        
        # Create processed image record
        processed_image = ProcessedImage(
            class_session_id=class_session_id,
            original_filename=original_filename,
            processed_filename=processed_filename,
            faces_detected=len(face_locations),
            faces_recognized=len(recognized_students)
        )
        db.session.add(processed_image)
        db.session.commit()
        
        return True, {
            "message": f"Successfully processed image with {len(recognized_students)} recognized students",
            "processed_image": processed_filename,
            "total_students": total_students,
            "present_students": present_students
        }
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error processing class photo: {str(e)}")
        return False, f"Error processing image: {str(e)}"

def process_multiple_photos(image_files, class_session_id):
    """Process multiple class photos for a single session"""
    try:
        # Get the class session
        class_session = ClassSession.query.get(class_session_id)
        if not class_session:
            return False, "Class session not found"
        
        # Load known faces
        known_face_encodings, known_student_ids, known_face_roll_numbers = load_known_faces()
        if not known_face_encodings:
            return False, "No registered student faces found"
        
        # Create directories if they don't exist
        uploads_folder = current_app.config['UPLOAD_FOLDER']
        processed_folder = current_app.config['PROCESSED_FOLDER']
        os.makedirs(uploads_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
        
        # Track overall results
        results = []
        present_student_ids = set()  # Use set to avoid duplicates
        total_faces_detected = 0
        total_faces_recognized = 0
        
        for image_file in image_files:
            try:
                # Load and process the class image
                original_filename = secure_filename(image_file.filename)
                image_path = os.path.join(uploads_folder, original_filename)
                
                # Save the uploaded file
                try:
                    image_file.save(image_path)
                except Exception as e:
                    logger.error(f"Failed to save uploaded file: {str(e)}")
                    results.append({
                        "filename": original_filename,
                        "faces_detected": 0,
                        "faces_recognized": 0,
                        "success": False,
                        "message": f"Failed to save uploaded file: {str(e)}"
                    })
                    continue
                
                # Load image for face recognition
                try:
                    image = face_recognition.load_image_file(image_path)
                    if image is None:
                        results.append({
                            "filename": original_filename,
                            "faces_detected": 0,
                            "faces_recognized": 0,
                            "success": False,
                            "message": "Failed to load the image"
                        })
                        continue
                except Exception as e:
                    logger.error(f"Error loading image for face recognition: {str(e)}")
                    results.append({
                        "filename": original_filename,
                        "faces_detected": 0,
                        "faces_recognized": 0,
                        "success": False,
                        "message": f"Failed to process image: {str(e)}"
                    })
                    continue
                
                face_locations = face_recognition.face_locations(image)
                total_faces_detected += len(face_locations)
                
                # If no faces detected, skip further processing
                if not face_locations:
                    results.append({
                        "filename": original_filename,
                        "faces_detected": 0,
                        "faces_recognized": 0,
                        "success": False,
                        "message": "No faces detected in the image"
                    })
                    continue
                    
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # For visualization, convert to BGR (OpenCV format)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Track recognized students for this image
                recognized_in_this_image = 0
            except Exception as e:
                logger.error(f"Error processing image {image_file.filename}: {str(e)}")
                results.append({
                    "filename": getattr(image_file, 'filename', 'unknown'),
                    "faces_detected": 0,
                    "faces_recognized": 0,
                    "success": False,
                    "message": f"Error: {str(e)}"
                })
                continue
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                # Find the best match
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        student_id = known_student_ids[best_match_index]
                        roll_number = known_face_roll_numbers[best_match_index]
                        
                        # Add to the set of present students
                        present_student_ids.add(student_id)
                        recognized_in_this_image += 1
                        total_faces_recognized += 1
                        
                        # Draw rectangle and student info on the image
                        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image_bgr, roll_number, (left + 6, bottom - 6), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        # Draw rectangle for unrecognized faces
                        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Save the processed image
            processed_filename = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}_{original_filename}"
            processed_path = os.path.join(processed_folder, processed_filename)
            
            # Save the processed image
            result = cv2.imwrite(processed_path, image_bgr)
            if not result:
                logger.error(f"Failed to save processed image to {processed_path}")
                results.append({
                    "filename": original_filename,
                    "faces_detected": len(face_locations),
                    "faces_recognized": recognized_in_this_image,
                    "success": False,
                    "message": "Failed to save processed image"
                })
                continue
            
            # Create processed image record
            processed_image = ProcessedImage(
                class_session_id=class_session_id,
                original_filename=original_filename,
                processed_filename=processed_filename,
                faces_detected=len(face_locations),
                faces_recognized=recognized_in_this_image
            )
            db.session.add(processed_image)
            
            # Add to results
            results.append({
                "filename": original_filename,
                "processed_filename": processed_filename,
                "faces_detected": len(face_locations),
                "faces_recognized": recognized_in_this_image,
                "success": True,
                "message": f"Successfully processed with {recognized_in_this_image} recognized students"
            })
        
        # Mark attendance for all recognized students
        for student_id in present_student_ids:
            attendance = Attendance.query.filter_by(
                student_id=student_id,
                class_session_id=class_session_id
            ).first()
            
            if not attendance:
                attendance = Attendance(
                    student_id=student_id,
                    class_session_id=class_session_id,
                    status='present'
                )
                db.session.add(attendance)
        
        # Update class session stats
        total_students = Student.query.count()
        present_students = len(present_student_ids)
        class_session.total_students = total_students
        class_session.present_students = present_students
        
        # Commit all changes
        db.session.commit()
        
        return True, {
            "results": results,
            "total_images": len(image_files),
            "total_faces_detected": total_faces_detected,
            "total_faces_recognized": total_faces_recognized,
            "present_students": present_students,
            "total_students": total_students
        }
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error processing multiple photos: {str(e)}")
        return False, f"Error processing images: {str(e)}"

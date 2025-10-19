document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const rollNumberSelect = document.getElementById('roll_number');
    const fileInput = document.getElementById('student_photo');
    const uploadForm = document.getElementById('uploadFaceForm');
    const imagePreview = document.getElementById('imagePreview');
    const faceData = document.getElementById('faceData');
    const faceLoader = document.getElementById('faceLoader');
    
    // Show image preview when a file is selected
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.innerHTML = `
                        <div class="card mb-3">
                            <div class="card-header">Image Preview</div>
                            <div class="card-body text-center">
                                <img src="${e.target.result}" class="img-fluid" alt="Preview">
                            </div>
                        </div>
                    `;
                };
                
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Load student faces when a roll number is selected
    if (rollNumberSelect) {
        rollNumberSelect.addEventListener('change', function() {
            const rollNumber = this.value;
            if (rollNumber) {
                loadStudentFaces(rollNumber);
            } else {
                faceData.innerHTML = '';
            }
        });
        
        // If a roll number is pre-selected, load the faces
        if (rollNumberSelect.value) {
            loadStudentFaces(rollNumberSelect.value);
        }
    }
    
    // Setup form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Basic client-side validation
            const rollNumber = rollNumberSelect.value;
            const photoFile = fileInput.files.length > 0;
            
            if (!rollNumber) {
                e.preventDefault();
                alert('Please select a student.');
                return false;
            }
            
            if (!photoFile) {
                e.preventDefault();
                alert('Please select a photo to upload.');
                return false;
            }
        });
    }
});

// Load faces for a specific student
function loadStudentFaces(rollNumber) {
    const faceData = document.getElementById('faceData');
    const faceLoader = document.getElementById('faceLoader');
    
    if (!faceData) return;
    
    // Show loader
    faceLoader.classList.remove('d-none');
    faceData.innerHTML = '';
    
    fetch(`/students/faces/${rollNumber}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load student faces');
            }
            return response.json();
        })
        .then(data => {
            faceLoader.classList.add('d-none');
            
            if (data.student && data.student.faces && data.student.faces.length > 0) {
                const student = data.student;
                
                // Create header
                const headerHtml = `
                    <div class="alert alert-info">
                        <h5 class="mb-0">Existing Face Images for ${student.name} (${student.roll_number})</h5>
                    </div>
                `;
                
                // Create face cards
                let facesHtml = '<div class="row">';
                
                student.faces.forEach(face => {
                    facesHtml += `
                        <div class="col-md-4 mb-3">
                            <div class="card h-100">
                                <img src="${face.image_url}" class="card-img-top" alt="Face Image">
                                <div class="card-body">
                                    <p class="card-text small text-muted">Added: ${face.created_at}</p>
                                    <button type="button" class="btn btn-sm btn-danger delete-face" 
                                            data-face-id="${face.id}" 
                                            data-bs-toggle="modal" 
                                            data-bs-target="#deleteModal">
                                        <i class="fas fa-trash-alt"></i> Delete
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                facesHtml += '</div>';
                faceData.innerHTML = headerHtml + facesHtml;
                
                // Setup delete buttons
                setupDeleteButtons();
            } else {
                faceData.innerHTML = `
                    <div class="alert alert-warning">
                        No face images found for this student. Upload the first one!
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            faceLoader.classList.add('d-none');
            faceData.innerHTML = `
                <div class="alert alert-danger">
                    Failed to load student faces. ${error.message}
                </div>
            `;
        });
}

// Setup delete buttons
function setupDeleteButtons() {
    const deleteButtons = document.querySelectorAll('.delete-face');
    const deleteFaceId = document.getElementById('deleteFaceId');
    
    deleteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const faceId = this.getAttribute('data-face-id');
            deleteFaceId.value = faceId;
        });
    });
    
    // Setup the delete confirmation form
    const deleteForm = document.getElementById('deleteFaceForm');
    if (deleteForm) {
        deleteForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const faceId = deleteFaceId.value;
            if (!faceId) return;
            
            fetch(`/students/face/${faceId}/delete`, {
                method: 'POST',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to delete face');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // Close the modal
                    const modal = bootstrap.Modal.getInstance(document.getElementById('deleteModal'));
                    modal.hide();
                    
                    // Reload the student faces
                    const rollNumberSelect = document.getElementById('roll_number');
                    loadStudentFaces(rollNumberSelect.value);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error deleting face: ' + error.message);
            });
        });
    }
}
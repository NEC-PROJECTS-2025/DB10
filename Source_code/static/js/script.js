// Function to validate contact form inputs
function validateContactForm(event) {
    event.preventDefault(); // Prevent form submission

    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const message = document.getElementById("message").value.trim();

    if (!name || !email || !message) {
        alert("Please fill in all the fields.");
        return false;
    }

    if (!validateEmail(email)) {
        alert("Please enter a valid email address.");
        return false;
    }

    alert("Form submitted successfully!");
    document.getElementById("contactForm").submit(); // Submit form if valid
}

// Email validation
function validateEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}

// Function for file upload validation in the prediction form
function validatePredictionForm(event) {
    const fileInput = document.getElementById("audioFile");
    if (!fileInput.value) {
        alert("Please upload an audio file.");
        event.preventDefault();
        return false;
    }

    const validExtensions = ["mp3", "wav"];
    const fileExtension = fileInput.value.split(".").pop().toLowerCase();
    if (!validExtensions.includes(fileExtension)) {
        alert("Only MP3 or WAV files are allowed.");
        event.preventDefault();
        return false;
    }
}

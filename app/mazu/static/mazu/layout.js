document.addEventListener('DOMContentLoaded', function() {
    // Get the modal
    const modal = document.getElementById("cookie-modal");

    // Get the <span> element that closes the modal
    const span = document.getElementsByClassName("close")[0]

    // Check localstorage cookie preference
    if (localStorage.getItem("cookies") == null) {
        localStorage.setItem("cookies", "undefined")
    }

    // Show modal if cookies not accepted yet
    if (localStorage.getItem("cookies") == "undefined") {
        modal.style.display = "block";
    }
    // Hide modal if cookies accepted
    else if (localStorage.getItem("cookies") == "accepted") {
        modal.style.display = "none";
    }
    
    // When user clicks on <span> (x), close the modal
    // and set localStorage "cookies" value to "accepted"
    span.onclick = function() {
        modal.style.display = "none";
        localStorage.setItem("cookies", "accepted");
    }
})
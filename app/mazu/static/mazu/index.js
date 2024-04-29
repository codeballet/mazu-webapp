document.addEventListener('DOMContentLoaded', function() {


    // Get csrf token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    const csrftoken = getCookie('csrftoken');


    // Check for when loader gets visible
    function respondToVisibility(element, callback) {
        const options = {
            root: document.documentElement, // The root element (usually the viewport)
        };
    
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                // Call the callback with a Boolean indicating visibility
                callback(entry.intersectionRatio > 0);
            });
        }, options);
    
        // Start observing the target element
        observer.observe(element);
    }


    respondToVisibility(document.querySelector(".loader"), () => {
        console.log("The loader class element is now visible!");

        let counter = 0

        // Start checking for an answer to prompt
        const check_answer = () => {
            fetch('http://localhost:8000/api_answer/', {
                method: 'get',
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);

                // After ten tries, reset the index page
                if (counter == 10) {
                    fetch('http://localhost:8000/api_reset/', {
                        method: 'post',
                        headers: {'X-CSRFToken': csrftoken},
                        mode: 'same-origin'
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log(data);
                        location.reload();
                    })

                }

                counter++;

                if (data.answer != '') {
                    location.reload();
                }
            });
        }

        setInterval(check_answer, 3000);

    });

})
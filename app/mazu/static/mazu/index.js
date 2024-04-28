document.addEventListener('DOMContentLoaded', function() {

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
        console.log("The element is now visible!");

        const check_answer = () => {
            fetch('http://localhost:8000/api_answer/', {
                method: "get",
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                if (data.answer != '') {
                    location.reload();
                }
            });
        }

        setInterval(check_answer, 2000);
    });

})
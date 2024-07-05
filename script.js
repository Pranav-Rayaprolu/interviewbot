document.addEventListener('DOMContentLoaded', function() {
    const totalQuestionsCircle = document.getElementById('total-questions-circle');
    const accuracyCircle = document.getElementById('accuracy-circle');
    const depthCircle = document.getElementById('depth-circle');
    const clarityCircle = document.getElementById('clarity-circle');
    const relevanceCircle = document.getElementById('relevance-circle');

    function setProgress(circle, value) {
        const radius = circle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        circle.style.strokeDasharray = `${circumference} ${circumference}`;
        circle.style.strokeDashoffset = circumference;

        const offset = circumference - (value / 100 * circumference);
        circle.style.strokeDashoffset = offset;
    }

    setProgress(totalQuestionsCircle, 10); // Example value for total questions
    setProgress(accuracyCircle, 80); // Example value for accuracy
    setProgress(depthCircle, 60); // Example value for depth
    setProgress(clarityCircle, 75); // Example value for clarity
    setProgress(relevanceCircle, 90); // Example value for relevance
});

async function saveLandmarks(label) {
    try {
        const response = await fetch('/save_landmarks', {
            method: 'POST',
            body: JSON.stringify({
                label: label,
            }),
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();
        alert(result.message);
    } catch (error) {
        console.error("Error:", error);
    }
}




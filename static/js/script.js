async function getPrediction() {
    const symptomsInput = document.getElementById('symptoms');
    const resultElement = document.getElementById('result');
    const symptoms = symptomsInput.value.trim();
    
    if (!symptoms) {
        resultElement.textContent = 'Please enter your symptoms';
        resultElement.style.color = 'red';
        return;
    }
    
    resultElement.textContent = 'Analyzing symptoms...';
    resultElement.style.color = 'blue';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symptoms }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultElement.innerHTML = `
                <strong>Possible diagnosis:</strong> ${data.disease}<br>
                <strong>Confidence:</strong> ${data.confidence}%<br>
                <em>Note: This is not a medical diagnosis. Please consult a healthcare professional.</em>
            `;
            resultElement.style.color = 'green';
        } else {
            resultElement.textContent = `Error: ${data.error || 'Something went wrong'}`;
            resultElement.style.color = 'red';
        }
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
        resultElement.style.color = 'red';
        console.error('Error:', error);
    }
}

// Allow pressing Enter key to submit
document.getElementById('symptoms').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        getPrediction();
    }
});
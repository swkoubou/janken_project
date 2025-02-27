async function saveLandmarks(label) {
    try {
        const response = await fetch('/save_landmarks', {
            method: 'POST',
            body: JSON.stringify({
                label: label,
                landmarks: Array(42).fill(Math.random().toFixed(2)) // 仮の座標データ
            }),
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();
        alert(result.message);
    } catch (error) {
        console.error("Error:", error);
    }
}

let chart;  // グローバル変数として宣言

async function fetchLandmarks() {
    const ctx = document.getElementById('landmarksChart').getContext('2d');  // ctxの初期化を関数内に移動

    try {
        const response = await fetch('/get_landmarks');
        const data = await response.json();

        if (Object.keys(data).length === 0) {
            console.log("No data available");
            return;
        }
        console.log("Fetched data:", data);  // デバッグ用

        const labels = Object.keys(data);
        const datasets = labels.map(label => ({
            label: label,
            data: data[label].map(coords => ({
                x: parseFloat(coords[0]),
                y: parseFloat(coords[1])
            })),
            borderColor: label === "0" ? 'red' : label === "1" ? 'blue' : 'green',
            pointRadius: 5,
            fill: false,
            showLine: false
        }));

        if (chart) {
            chart.destroy(); // 既存グラフを破棄
        }

        chart = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                scales: {
                    x: { beginAtZero: true },
                    y: { beginAtZero: true }
                }
            }
        });
    } catch (error) {
        console.error("Failed to fetch landmarks:", error);
    }
}



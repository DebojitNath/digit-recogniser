// Pure Vanilla JavaScript Canvas Drawing with Mouse and Touch Support
document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;

    const context = canvas.getContext("2d");
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Fill canvas with solid black
    function clearCanvas() {
        context.fillStyle = "black";
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    clearCanvas();

    function getCoords(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let clientX, clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getCoords(e);
        lastX = pos.x;
        lastY = pos.y;

        // Draw a single dot in case the user just clicks/taps
        draw(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const pos = getCoords(e);

        context.beginPath();
        context.moveTo(lastX, lastY);
        context.lineTo(pos.x, pos.y);

        // MNIST styled thick white stroke with rounded caps
        context.strokeStyle = "white";
        context.lineWidth = 26;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.stroke();

        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDrawing(e) {
        if (isDrawing) {
            isDrawing = false;
        }
    }

    // Mouse Event Listeners
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    // Touch Event Listeners (Mobile & Tablet)
    canvas.addEventListener("touchstart", startDrawing, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDrawing, { passive: false });
    canvas.addEventListener("touchcancel", stopDrawing, { passive: false });
});

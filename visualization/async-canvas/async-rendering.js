// A cooperative-"multithreading" JavaScript routine to allow asynchronous rendering and
// display without the cost of WebWorker serialization.                -- @CasualEffects

var updateImageData = ctx.createImageData(screenWidth, screenHeight);
var updateImage = document.createElement('canvas');
updateImage.width = screenWidth;
updateImage.height = screenHeight;

var makeRenderLoop = function*() {
    while (true) {
        // Pretend you have actual rendering code here...
        var x = Math.floor(Math.random() * screenWidth);
        var y = Math.floor(Math.random() * screenHeight);

        var r = 255;
        var g = 0;
        var b = Math.floor(Math.random() * 255);

        // write to a pixel
        var pix = (x + y * screenWidth) * 4;
        updateImageData.data[pix + 0] = r;
        updateImageData.data[pix + 1] = g;
        updateImageData.data[pix + 2] = b;
        updateImageData.data[pix + 3] = 255;
        
        yield;
    }
};

var coroutine = makeRenderLoop();

function drawFrame(time) {
    // Run the "infinite" loop for a while
    for (var i = 0; i < 100; ++i) {
        coroutine.next();
    }
    
    // Update the image
    updateImage.getContext('2d').putImageData(updateImageData, 0, 0);
    ctx.drawImage(updateImage, 0, 0, screen.width, screen.height);

    // Keep the callback chain going
    requestAnimationFrame(drawFrame);
}

requestAnimationFrame(drawFrame);

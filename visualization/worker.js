// waiting to receive the message
self.onmessage = (event) => {
    const mat = event.data; // a matrix in exp_matrices
    const matName = mat.mat_name;
    const matSparse = mat.mat_sparse;
    const matPath = mat.mat_path;
    const matColorR = mat.mat_color[0];
    const matColorG = mat.mat_color[1];
    const matColorB = mat.mat_color[2];
    const matRows = mat.mat_shape[0];
    const matCols = mat.mat_shape[1];
    
    var osc = new OffscreenCanvas(matCols, matRows); // offscreen canvas
    var ctx = osc.getContext("2d"); // canvas context

    // read csv
    matData = [];
    fetch(matPath) // relative path only
        .then(response => response.text())
        .then(csvData => {
            const lines = csvData.trim().split('\n');
            for (const line of lines) {
                const values = line.trim().split(',');
                matData.push(values);
            }
        })
        .catch(error => {
            console.error('Error:', error);
    });

    // set canvas
    ctx.clearRect(0, 0, osc.width, osc.height);
    if (matSparse === true) { // COO
        for (const row in matData) {
            const r = row[0];
            const c = row[1];
            const v = row[2];
            const t = 0.5;
            ctx.fillStyle = 'rgba(${matColorR},${matColorG},${matColorB},${t})';
            ctx.fillRect(c, r, 1, 1);
        }
    }
    else {
        for (var r = 0; r < matRows; r++) {
            for (var c = 0; c < matCols; c++) {
                const v = matData[r][c];
                if (v == 0) {
                    continue;
                }
                ctx.fillStyle = 'rgba(${matColorR},${matColorG},${matColorB},${t})';
                ctx.fillRect(c, r, 1, 1);
            }
        }
    }  
    postMessage([matName, osc]);
};

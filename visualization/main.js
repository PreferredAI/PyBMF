var inputZoom = document.getElementById("inputZoom");
var checkboxVerbose = document.getElementById("checkboxVerbose");

var settingsObj = null; // settings json
var expFramesObj = null; // exp_frames json
var expMatricesObj = null; // exp_matrices json
var frameCellsObj = null; // frame_cells json

var frameTable = document.getElementById("frameTable");

var expList = document.getElementById("expList");
var frameList = document.getElementById("frameList");
var expComment = document.getElementById("expComment");
var frameComment = document.getElementById("frameComment");

const worker = new Worker("worker.js");
var oscList = null; // offscreen canvas list

function readFile() {
    // manually read json file
    const [file] = document.querySelector("input[type=file]").files;
    const reader = new FileReader();
    reader.addEventListener("load", () => {
            settingsObj = JSON.parse(reader.result);
            setExpList();
        },
        false
    );
    if (file) {
        reader.readAsText(file);
    }
}

function readFileAuto() {
    // automatically read json file upon loading
    fetch('settings.json')
        .then(response => response.json())
        .then(jsonResponse => {
            settingsObj = jsonResponse;
            setExpList();
        })
        .catch((e) => console.error(e));
}

function readCSV(csvPath, sep=',') {
    // read csv file
    fetch(csvPath)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`Failed to fetch ${filePath}: ${response.status} ${response.statusText}`);
            }
            return response.text();
        })
        .then((csv) => {
            const rows = csv.split(sep);
            const matrix = rows.map((row) => row.split(',').map(Number));
            saveMatrixToCanvas(matrix);
        })
        .catch((e) => console.error(e));
}

window.onload = (event) => {
    readFileAuto();
};

function setExpList() {
    // trigerred by readFile() or readFileAuto()
    // reset expList
    expList.innerHTML = "<option> ---Choose experiment--- </option>";
    // add experiments to expList
    for (const exp of settingsObj.experiments) {
        const option = document.createElement("option");
        option.text = exp.exp_name;
        option.setAttribute("value", exp.exp_name);
        expList.appendChild(option);
    }
}

function setExp() {
    // trigerred by the selection upon experiments
    const expName = expList.options[expList.selectedIndex].text;
    // reset frameList
    frameList.innerHTML = "<option> ---Choose frame--- </option>";
    // locate experiment
    for (const exp of settingsObj.experiments) {
        if (exp.exp_name == expName) {
            
            // set expMatrices
            expMatricesObj = exp.exp_matrices;
            setMatrixList();
            // set expFrames
            expFramesObj = exp.exp_frames;
            setFrameList();
            // set expComment (must come after setMatrixList())
            expComment.innerText = exp.exp_comment;
            // set expZoom
            inputZoom.value = exp.exp_zoom;
            break;
        }
    }
}

function setMatrixList() {
    // triggered by setExp()
    // reset offscreenCanvasList
    oscList = new Map();
    // load Map object
    for (const mat of expMatricesObj) {
        // mat.mat_path = settingsObj.root.concat(mat.mat_path)
        // send to the worker thread
        worker.postMessage(mat); // post message
        worker.onmessage = function(event) { // receive message
            const matName = event.data[0];
            const offscreenCanvas = event.data[1]
            oscList.set(matName, offscreenCanvas)
            console.log('Cached matName: ' + matName);
        }
    }
}

function setFrameList() {
    // triggered by setExp()
    // reset frameList
    frameList.innerHTML = "<option> ---Choose frame--- </option>";
    // add frames to frameList
    for (const frame of expFramesObj) {
        const option = document.createElement("option");
        option.text = frame.frame_name
        option.setAttribute("value", frame.frame_name);
        frameList.appendChild(option);
    }
}

function setFrame() {
    // trigerred by the selection upon frames
    const frameName = frameList.options[frameList.selectedIndex].text;
    // locate frame
    for (const frame of expFramesObj) {
        if (frame.frame_name == frameName) {
            // set frameComment
            frameComment.innerText = frame.frame_comment;
            // set frameTable
            frameCellsObj = frame.frame_cells;
            setFrameTable(frame.frame_shape);
        }
    }
}

function setFrameTable(frameShape) {
    // triggered by setFrame()
    // reset frameTable
    frameTable.innerHTML = "";
    tableRows = frameShape[0];
    tebleCols = frameShape[1];
    for (var r = 0; r < tableRows; r++) {
        const row = frameTable.insertRow();
        for (var c = 0; c < tebleCols; c++) {
            const cell = row.insertCell();
            row.appendChild(cell);
        }
    }
    // load cells onto frameTable
    for (const cell of frameCellsObj) {
        r = cell.cell_location[0];
        c = cell.cell_location[1];
        // locate new cell in frameTable
        const newCell = frameTable.rows[r].cells[c]
        // create new div for stacked canvas
        const newDiv = document.createElement("div");
        let zIndex = 0;
        for (const matName of frameCellsObj.cell_matrices) {
            zIndex++;
            const newCanvas = document.createElement("canvas");
            newCanvas.getContext("2d").drawImage(oscList.get(matName), 0, 0); // get offscreen canvas
            newCanvas.setAttribute("style", "z-index: ${zIndex}; position:relative;");
            newDiv.appendChild(newCanvas);
        }
        newCell.appendChild(newDiv);
        if (checkboxVerbose.checked) {
            const cellName = document.createTextNode(cell.cell_name);
            newCell.appendChild(cellName);
        }
    }
}

// function clickRefresh() {	
//     const frameIndex = frameList.selectedIndex;
//     setPage(refresh = true);
//     frameList.selectedIndex = frameIndex;
//     setFrame();
// }

function gotoFrame(action) {
    const frameIndex = frameList.selectedIndex;
    if (frameIndex < 1) {
        return 0;
    }
    if (action == 0 && frameIndex > 1) {
        frameList.selectedIndex = frameIndex - 1;
        setFrame();
    }
    if (action == 1 && frameIndex < frameList.options.length - 1) {
        frameList.selectedIndex = frameIndex + 1;
        setFrame();
    }
}
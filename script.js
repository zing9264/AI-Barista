

var lock='unlock';

var btn1 = document.getElementById("btn1");
var btn2 = document.getElementById("btn2");
var btn3 = document.getElementById("btn3");

var ws = new WebSocket('ws://localhost:8080');

switch (ws.readyState) {
    case WebSocket.CONNECTING:
        // do something
        console.log("CONNECTING");
        break;
    case WebSocket.OPEN:
        // do something
        console.log("OPEN");
        break;
    case WebSocket.CLOSING:
        console.log("CLOSING");

        // do something
        break;
    case WebSocket.CLOSED:
        // do something
        console.log("CLOSED");

        break;
    default:
        // this never happens
        break;
}

ws.onopen = function () {
    ws.send('Hello Server!');
};
ws.onclose = function(event) {
    var code = event.code;
    var reason = event.reason;
    var wasClean = event.wasClean;
    // handle close event
};

ws.addEventListener("close", function(event) {
    var code = event.code;
    var reason = event.reason;
    var wasClean = event.wasClean;
    // handle close event
});

ws.onmessage = function(event){
    console.log(typeof event.data );
    console.log(event.data );
    if(event.data=="locking" ||event.data=="unlock"){
        lock=event.data;
    }
};


function btnclicked(event) {
    console.log(event.srcElement.id);
    ws.send(event.srcElement.id);
}

btn1.addEventListener("click",btnclicked);
btn2.addEventListener("click",btnclicked);
btn3.addEventListener("click",btnclicked);

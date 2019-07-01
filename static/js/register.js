class Register{
    constructor(containerElement){
        this.containerElement = containerElement;
        this.backToMenuContainer = containerElement.querySelector(".to-RegOrLog");
        this.backToMenuContainer.addEventListener('click', this.backtoLastPage);
        const formElement = containerElement.querySelector("form");
        formElement.addEventListener("submit", function(event){
            event.preventDefault();
            const ra = document.querySelector("#ra");
            const rp = document.querySelector("#rp");
            const rcp = document.querySelector("#rpc");
            var arr = [ra.value,rp.value, "0"];
            if(rp.value != rcp.value){
                const tmp = document.querySelector("#regresult");
                tmp.innerHTML="兩次密碼不同，請重新輸入";
            }
            else{
                fetch('/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(arr)
                }).then(function(res){
                    return res.json();
                }).then(function(js){
                    if(js.response == "success"){
                        app.regResult.classList.remove("inactive");
                        app.register.hide();
                        const backToMenuContainer = app.regResult.querySelector(".to-menu");
                        backToMenuContainer.addEventListener('click', app.register.backtoMenu);
                //        span.classList.remove("inactive");
                    }
                    else
                        console.log(js.response);
                })
            }

        });
    }
    backtoMenu() {
      app.register.hide();
      app.regResult.classList.add("inactive");
      app.menu.show();
    }
    backtoLastPage() {
      app.regorlog.show();
      app.register.hide();
      document.getElementById("google").innerHTML="0";
    }

    show(){
        this.containerElement.classList.remove("inactive");
    }

    hide(){
        this.containerElement.classList.add("inactive");
    }
}

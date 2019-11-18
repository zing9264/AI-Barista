class Login{
    constructor(containerElement){
        this.containerElement = containerElement;
        this.backToMenuContainer = containerElement.querySelector(".to-RegOrLog");
        this.backToMenuContainer.addEventListener('click', this.backtoLastPage);
        const formElement = containerElement.querySelector("form");
        formElement.addEventListener("submit", function(event){
            event.preventDefault();
            const la = document.querySelector("#la");
            const lp = document.querySelector("#lp");
            var arr = [la.value, lp.value];
            console.log(arr);
            fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(arr)
            }).then(function(res){
                return res.json();
            }).then(function(js){
                if(js.response == "success"){
                    app.login.hide();
                    app.loginsuc.classList.remove("inactive");
                    const header = document.querySelector("header");
              //      header.querySelector("section").classList.add("inactive");
              //      const span = header.querySelector("span");
                    document.getElementById("Score").innerHTML=`${js.Score}`;
              //      span.classList.remove("inactive");

                    const backToMenuContainer = app.loginsuc.querySelector(".to-menu");
                    backToMenuContainer.addEventListener('click', app.login.backtoMenu);

                  //  app.main.show();
              //      app.login.hide();
                }
                else{
                    document.querySelector("#loginerr").classList.remove("inactive");
                }
            })
        });
    }
    backtoMenu() {
      app.register.hide();
      app.regResult.classList.add("inactive");
      app.menu.show();
    }
    backtoLastPage() {
      app.regorlog.show();
      app.login.hide();
      document.getElementById("google").innerHTML="0";
    }
    show(){
        this.containerElement.classList.remove("inactive");
    }

    hide(){
        this.containerElement.classList.add("inactive");
    }
}

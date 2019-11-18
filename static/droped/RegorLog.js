class RegorLog{
  constructor(containerElement){
    this.containerElement = containerElement;
    this.ToRegContainer = containerElement.querySelector(".to-register");
    this.ToLogContainer = containerElement.querySelector(".to-login");
    this.backToMenuContainer = containerElement.querySelector(".to-menu");
    this.backToMenuContainer.addEventListener('click', this.backtoMenu);
    this.ToRegContainer.addEventListener('click', this.toReg);
    this.ToLogContainer.addEventListener('click', this.toLog);
  }
  toReg() {
    app.regorlog.hide();
    app.register.show();
    document.getElementById("google").innerHTML="1";
  }
  toLog() {
    app.regorlog.hide();
    app.login.show();
    document.getElementById("google").innerHTML="2";
  }
  backtoMenu() {
    app.regorlog.hide();
    app.menu.show();
  }
  show() {
    this.containerElement.classList.remove('inactive');
  }

  hide() {
    this.containerElement.classList.add('inactive');
  }
}

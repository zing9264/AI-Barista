class MachineArm{
  constructor(containerElement){
    this.containerElement = containerElement;
    this.loading = this.loading.bind(this);
    this.backToMenuContainer = containerElement.querySelector(".to-menu");
    this.backToMenuContainer.addEventListener('click', this.backtoMenu);
    this.submitbtn=containerElement.querySelector("#submitbtn");
    this.submitbtn.addEventListener('click', this.loading);


  }
  backtoMenu() {
    window.location.href="/";
  }
  reUpload(){
    window.location.href="/upload";
  }

  
  loading() {
    this.containerElement.classList.add('inactive');
    document.querySelector("#loading").classList.remove('inactive');
    document.getElementById("form").submit()
  }

  show() {
    this.containerElement.classList.remove('inactive');
  }
  hide() {
    this.containerElement.classList.add('inactive');
  }
}

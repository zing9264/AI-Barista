class App {
  constructor() {
    const menuElement = document.querySelector('#menu');
    this.menu = new MenuScreen(menuElement);

    const armElement = document.querySelector('#arm');
    this.machinearm = new MachineArm(armElement);

    const introduceElement = document.querySelector('#introduce');
    this.introduce = new Introduction(introduceElement);

//    const quizElement = document.querySelector('#quiz');
  //  this.quiz = new Quiz(quizElement);

    const regorlogElement = document.querySelector("#RegOrLog");
    this.regorlog = new RegorLog(regorlogElement);

    const registerElement = document.querySelector("#register");
    this.register = new Register(registerElement);

    const regResultElement = document.querySelector("#reg_result");
    this.regResult = regResultElement;

    const loginElement = document.querySelector("#login");
    this.login = new Login(loginElement);

    const loginsucElement = document.querySelector("#loginsuc");
    this.loginsuc = loginsucElement;

    const demovideoElement = document.querySelector('#demovideo');
    this.demovideo = new DemoVideo(demovideoElement);

//    const resultElement = document.querySelector('#results');
//    this.results = new ResultsScreen(resultElement);

//    const storesucElement = document.querySelector("#storesuc");
//    this.storesuc = storesucElement;

  }
}

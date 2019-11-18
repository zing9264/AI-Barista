class App {
  constructor() {
    const menuElement = document.querySelector('#menu');
    this.menu = new MenuScreen(menuElement);

    const armElement = document.querySelector('#arm');
    this.machinearm = new MachineArm(armElement);

    const introduceElement = document.querySelector('#introduce');
    this.introduce = new Introduction(introduceElement);

    const demovideoElement = document.querySelector('#demovideo');
    this.demovideo = new DemoVideo(demovideoElement);


  }
}

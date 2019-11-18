class App {
  constructor() {
    const armElement = document.querySelector('#arm');
    this.machinearm = new MachineArm(armElement);
  }
}

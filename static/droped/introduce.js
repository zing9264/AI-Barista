class Introduction{
  constructor(containerElement){
    this.containerElement = containerElement;
    this.backToMenuContainer = containerElement.querySelector(".to-menu");
    this.ToDemovideoContainer = containerElement.querySelector(".to-demo");
    this.backToMenuContainer.addEventListener('click', this.backtoMenu);
    this.ToDemovideoContainer.addEventListener('click', this.toDemo);
  }
  toDemo() {
    app.introduce.hide();
    app.demovideo.show();
  }
  backtoMenu() {
    app.introduce.hide();
    app.menu.show();
  }
  show() {
    document.body.style.backgroundImage = "url('http://www.wendywl.uk/blog/wp-content/uploads/2017/08/hM8bEk9yc4tfGoTp5aheU2u01B3Bn9FJyI8ZRtgE.jpeg')";
    this.containerElement.classList.remove('inactive');
  }

  hide() {
    this.containerElement.classList.add('inactive');
    document.body.style.backgroundImage = "url('https://www.reviewgeek.com/thumbcache/2/200/40d2d1911ab4c74c596d316902194b0a/p/uploads/2018/08/1b6b0fd0.jpg')";
  }
}

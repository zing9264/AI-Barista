class DemoVideo{
  constructor(containerElement){
    this.containerElement = containerElement;
    this.backToMenuContainer = containerElement.querySelector(".to-menu");
    this.backToMenuContainer.addEventListener('click', this.backtoMenu);
  }
  backtoMenu() {
    app.demovideo.hide();
    app.menu.show();
  }
  show() {
    document.body.style.backgroundImage = "url('http://christianbackgrounds.info/new_images/25/63755653-coffee-wallpapers.jpg')";
    this.containerElement.classList.remove('inactive');
  }

  hide() {
    this.containerElement.classList.add('inactive');
    document.body.style.backgroundImage = "url('https://www.reviewgeek.com/thumbcache/2/200/40d2d1911ab4c74c596d316902194b0a/p/uploads/2018/08/1b6b0fd0.jpg')";
  }
}

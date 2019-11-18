// TODO(you): Modify the class in whatever ways necessary to implement
// the flashcard app behavior.
//
// You may need to do things such as:
// - Changing the constructor parameters
// - Adding methods
// - Adding additional fields

class MenuScreen {
  constructor(containerElement) {
    this.containerElement = containerElement;
    for (let i = 0; i < FLASHCARD_DECKS.length; i ++) {
    	let div = document.createElement("div");
    	div.className = "menu-buttons";
    	div.appendChild(document.createTextNode(FLASHCARD_DECKS[i].title));
    	div.addEventListener('click', function() {
      if(i == 0){
        app.menu.hide();
        app.introduce.show();
      }
      else if(i == 1) {
        //app.menu.hide();
       // app.machinearm.show();
       window.location.href='/upload'
      }
      else if(i == 2){
        app.menu.hide();
        app.demovideo.show();
      }
      else if(i == 3){
        app.menu.hide();
        app.regorlog.show();
      }
    	},false);
    	document.getElementById("choices").appendChild(div);
    }
  }

  show() {
    this.containerElement.classList.remove('inactive');
  }

  hide() {
    this.containerElement.classList.add('inactive');
  }
}

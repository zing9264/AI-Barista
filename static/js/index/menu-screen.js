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
    for (let i = 0; i < Menu_Title.length; i ++) {
    	let div = document.createElement("div");
    	div.className = "push_button blue";
    	div.appendChild(document.createTextNode(Menu_Title[i].title));
    	div.addEventListener('click', function() {
      if(i == 0){
       window.location.href='/introduce'
      }
      else if(i == 1) {
       window.location.href='/upload'
      }
      else if (i == 2) {
        window.location.href='/demovideo' 
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

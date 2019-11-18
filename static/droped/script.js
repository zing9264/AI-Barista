function onSignIn(googleUser) {
  var profile = googleUser.getBasicProfile();
  console.log(document.getElementById("google").innerHTML);

  if(document.getElementById("google").innerHTML==1){
    document.getElementById("ra").setAttribute('value',profile.getName());
    document.getElementById("rp").setAttribute('value',profile.getId());
    var arr = [ra.value,rp.value, "0"];
    app.regResult.classList.remove("inactive");
    app.register.hide();
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(arr)
        }).then(function(res){
          return res.json();
        }).then(function(js){
          if(js.response == "success"){
            app.regResult.classList.remove("inactive");
            app.register.hide();
            const backToMenuContainer = app.regResult.querySelector(".to-menu");
            backToMenuContainer.addEventListener('click', app.register.backtoMenu);
  //          span.classList.remove("inactive");
          }
          else
            console.log(js.response);
          })
          document.getElementById("google").innerHTML="0";
  }
  else if(document.getElementById("google").innerHTML==2){
    document.getElementById("la").setAttribute('value',profile.getName());
    document.getElementById("lp").setAttribute('value',profile.getId());

    var arr = [la.value, lp.value];
    
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

          document.getElementById("Score").innerHTML=`${js.Score}`;

          const backToMenuContainer = app.loginsuc.querySelector(".to-menu");
          backToMenuContainer.addEventListener('click', app.login.backtoMenu);
        }
        else{
            document.querySelector("#loginerr").classList.remove("inactive");
        }
    })
    document.getElementById("google").innerHTML="0";
  }
  console.log('ID: ' + profile.getId()); // Do not send to your backend! Use an ID token instead.
  console.log('Name: ' + profile.getName());
  console.log('Image URL: ' + profile.getImageUrl());
  console.log('Email: ' + profile.getEmail()); // This is null if the 'email' scope is not present.
}
function signOut() {
    var auth2 = gapi.auth2.getAuthInstance();
    auth2.signOut().then(function () {
      console.log('User signed out.');
    });
  }

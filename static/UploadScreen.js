class UploadScreen {
    constructor(screenElement){

    }
}








function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            $('#blah').attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
        console.log(reader)
    }
}

$("#imgInp").change(function() {
    readURL(this);
});

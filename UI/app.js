// The 'Dropzone.autoDiscover' property is used to
//enable or disable the automatic discovery feature,
// which searches the document for elements with the
// 'dropzone' class and automatically turns them into
// drop zones. By setting it to 'false', the automatic
// discovery feature is turned off, and the drop zones
// need to be explicitly initialized in the JavaScript code.
Dropzone.autoDiscover = false;

//initializing a Dropzone instance
//The Dropzone object is created by calling the constructor 'Dropzone' with two arguments: the first argument is a selector for the element that will be used as the drop zone (in this case, it's a element with the ID 'dropzone'), and the second argument is an object literal that contains the configuration options for the drop zone.
//The options specified in this configuration include:
//
//'url': the URL to which the file will be uploaded. In this case, the URL is set to '/', which means the file will be uploaded to the root of the current domain.
//
//'maxFiles': the maximum number of files that can be uploaded at once. In this case, it is set to 1, meaning only one file can be uploaded at a time.
//
//'addRemoveLinks': a Boolean value that determines whether to show links to remove individual files from the drop zone. In this case, it is set to 'true', which means the remove links will be shown.
//
//'dictDefaultMessage': the default message that is displayed in the drop zone when there are no files. In this case, it is set to "Some Message".
//
//'autoProcessQueue': a Boolean value that determines whether to automatically upload the files as soon as they are added to the drop zone. In this case, it is set to 'false', which means the files will not be automatically uploaded, and they will need to be manually uploaded using the drop zone's methods.
function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

// defines an event handler for the 'addedfile' event of the Dropzone object 'dz'.
// The event handler is executed every time a file is added to the drop zone.

// This function checks if the second file exists in the 'dz.files' array,
// which holds all the files that have been added to the drop zone.
// If the second file exists, it means the maximum number of
// files (1 in this case) has been exceeded, so the first file
// is removed using the 'removeFile' method of the drop zone.
// This ensures that only the maximum number of files specified
// in the 'maxFiles' option (1 in this case) is present in the drop zone at any given time.
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);
        }
    });


//defines an event handler for the 'complete' event of the
// Dropzone object 'dz'. The event handler is executed
// every time a file has finished uploading.
    dz.on("complete", function (file) {
//        The function takes the uploaded file object 'file' as an argument and retrieves the 'dataURL'
// property of the file, which is the base64 encoded data of the uploaded image.
        let imageData = file.dataURL;
        
        var url = "http://127.0.0.1:5000/classify_image";
//      It then makes a POST request to the specified URL 'http://127.0.0.1:5000/classify_image'
// with the base64 encoded image data as a form data 'image_data'. The function passed as a
// second argument to the $.post() method is executed as a callback when the request is complete.
        $.post(url, {
            image_data: file.dataURL
        },function(data, status) {

//          In the callback function, the data returned from the server is logged
// in the console using 'console.log(data)'. If the data is empty or undefined,
// the 'error' div is shown and the 'resultHolder' and 'divClassTable' divs are hidden.
            console.log(data[0]);

            if (!data || data.length==0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();                
                $("#error").show();
                return;
            }
            let players = ["beckham", "mbappe", "messi", "neymar", "salah"];

let match = data[0]
console.log(match)
let classProbability = match["class_probability"];
console.log(classProbability);

//            If there is a match, the 'error' div is hidden
// and the 'resultHolder' and 'divClassTable' divs are shown.
// The content of the 'resultHolder' div is updated with the HTML
// content of a div that has a 'data-player' attribute equal to the matched class.
            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                $("#resultHolder").html($(`[data-player="${match.class}"`).html());
//                let classDictionary = match.class_dictionary;


            }
            // dz.removeFile(file);            
        });
    });


// This code defines an event handler for the submit button with an ID of "submitBtn".
// When the button is clicked, the function is executed and the method "processQueue()"
// is called on the Dropzone object (dz). This will start the process of uploading the selected file to the server.
    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}



// This code sets up the document ready event handler using jQuery.
// When the page has finished loading, the anonymous function inside
// the event handler will be executed. The code inside the event handler
// performs the following actions:

//Hides the elements with the IDs of "error", "resultHolder", and "divClassTable".

//Calls the "init()" function.

//This code ensures that the page is properly initialized when it is loaded and ready to use.
$(document).ready(function() {
    console.log( "ready!" );
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});
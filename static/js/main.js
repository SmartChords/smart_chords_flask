function addFileName() {
  const fileInput = document.querySelector('#fileToUpload');
  const fileName = document.querySelector('#fileName');
  if (fileInput.files.length > 0) {
    fileName.innerText = fileInput.files[0].name;
  } else {
    fileName.innerText = 'Choose file';
  }
}

function sendToPreview() {
  event.preventDefault();
  var form = document.getElementById('preview-image-form');
  var formData = new FormData(form);
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/predict', true);
  xhr.send(formData);
  document.querySelector('#preview-block').style.display = 'none';
  document.querySelector('#loading-block').style.display = '';
  xhr.onreadystatechange = function() { // Call a function when the state changes.
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
      console.log(xhr.response)
      // window.location.pathname = "annotated/"
    }
}
}

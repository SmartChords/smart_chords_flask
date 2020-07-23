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
  xhr.onreadystatechange = function() {
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
      document.querySelector('#loading-block').style.display = 'none';
      var annotation = document.querySelector('#annotated-block');
      document.querySelector('#download-link-option').style.display = '';
      annotation.innerHTML = this.responseText;
    } else if (this.status === 500) {
      window.location.pathname = '/error';
  	}
  }
}

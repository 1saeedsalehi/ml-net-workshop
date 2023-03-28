
const serviceUrl = 'api/ImageClassification/classifyImage';
const form = document.querySelector('form');

form.addEventListener('submit', e => {
    e.preventDefault();

    const files = document.querySelector('[type=file]').files;
    const formData = new FormData();

    formData.append('imageFile', files[0]);


    fetch(serviceUrl, {
        method: 'POST',
        body: formData
    }).then((resp) => resp.json())
      .then(function (response) {
          console.info('Response', response);
          console.log('Response', response);

          debugger;
          document.getElementById('divPrediction').innerHTML = "it's " + (response.probability * 100).toFixed(3) + "% " + response.predictedLabel;

          return response;
        });


});
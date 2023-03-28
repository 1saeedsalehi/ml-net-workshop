using ImageClassification.DataModels;
using ImageClassification.WebApp.ImageHelpers;
using ImageClassification.WebApp.ML.DataModels;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace TensorFlowImageClassification.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageClassificationController : ControllerBase
    {
        public IConfiguration Configuration { get; }
        private readonly PredictionEnginePool<InMemoryImageData, ImagePrediction> _predictionEnginePool;
        private readonly ILogger<ImageClassificationController> _logger;

        public ImageClassificationController(
            PredictionEnginePool<InMemoryImageData, ImagePrediction> predictionEnginePool,
            IConfiguration configuration,
            ILogger<ImageClassificationController> logger) 
        {
            // Get the ML Model Engine injected, for scoring.
            _predictionEnginePool = predictionEnginePool;

            Configuration = configuration;

            // Get other injected dependencies.
            _logger = logger;
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("classifyImage")]
        public async Task<IActionResult> ClassifyImage(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return BadRequest();

            var imageMemoryStream = new MemoryStream();
            await imageFile.CopyToAsync(imageMemoryStream);

            //TODO: Check that the image is valid.
            byte[] imageData = imageMemoryStream.ToArray();
            

            _logger.LogInformation("Start processing image...");

          
            // Set the specific image data into the ImageInputData type used in the DataView.
            var imageInputData = new InMemoryImageData(image: imageData, label: null, imageFileName: null);

            // Predict code for provided image.
            var prediction = _predictionEnginePool.Predict(imageInputData);

      
            // Predict the image's label (The one with highest probability).
            var imageBestLabelPrediction =
                new ImagePredictedLabelWithProbability
                {  
                    PredictedLabel = prediction.PredictedLabel,
                    Probability = prediction.Score.Max()
                };

            return Ok(imageBestLabelPrediction);
        }

    }
}
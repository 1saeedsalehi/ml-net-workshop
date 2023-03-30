using Microsoft.ML.Data;

namespace MovieRecommender_Model.Model;

public class MovieRating
{
    [LoadColumn(0)]
    public string userId;

    [LoadColumn(1)]
    public string movieId;

    [LoadColumn(2)]
    public bool Label;
}

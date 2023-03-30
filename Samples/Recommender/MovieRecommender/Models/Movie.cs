using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MovieRecommender.DataStructures
{
    class Movie
    {
        public int movieId;

        public String movieTitle;

        private static String moviesdatasetRelativepath = $"{Program.DatasetsRelativePath}/recommendation-movies.csv";
        private static string moviesdatasetpath = Program.GetAbsolutePath(moviesdatasetRelativepath);

        public Lazy<List<Movie>> _movies = new(() => LoadMovieData(moviesdatasetpath));

        public Movie()
        {
        }

        public Movie Get(int id)
        {
            return _movies.Value.Single(m => m.movieId == id);
        }

        private static List<Movie> LoadMovieData(string moviesdatasetpath)
        {
            var result = new List<Movie>();
            Stream fileReader = File.OpenRead(moviesdatasetpath);
            StreamReader reader = new(fileReader);
            try
            {
                bool header = true;
                int index = 0;
                var line = "";
                while (!reader.EndOfStream)
                {
                    if (header)
                    {
                        line = reader.ReadLine();
                        header = false;
                    }
                    line = reader.ReadLine();
                    string[] fields = line.Split(',');
                    int movieId = int.Parse(fields[0].ToString());
                    string movieTitle = fields[1].ToString();
                    result.Add(new Movie() { movieId = movieId, movieTitle = movieTitle });
                    index++;
                }
            }
            finally
            {
                reader?.Dispose();
            }

            return result;
        }
    }
}

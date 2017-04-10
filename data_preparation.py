def create_expanded_movie_genres_dataframe(movies, genres, genreColumnsPrefix='isgenre:', 
                                           auxColName='auxCol', deleteGenresColumn=True): 
    
    # (1) Create a copy of the dataframe.
    movies2 = movies.copy()
    
    # (2) Create a temporary column in the dataframe, 
    # containing the split of the genres column.
    movies2[auxColName] = movies2.genres.str.split('|')
    
    # (3) Assign True to each Boolean column if the 
    # genre title exists in the auxiliary column.
    for g in genres: 
        n = genreColumnsPrefix + g
        movies2[n] = movies2.auxCol.map(lambda x: g in x)
    
    del movies2[auxColName]
    if deleteGenresColumn: 
        del movies2['genres']
        
    return movies2
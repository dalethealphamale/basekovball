# basekovball
Markov Chain Model of Baseball for Batting Order Optimization

A Markov Chain simulation of a Baseball game, which outputs the expected number of runs scored per game for a given lineup, and can be used for batting order optimization or to compare the expected scoring output of various lineups.

This code revists "Markov Chain Models for Baseball.pdf", a senior thesis written and presented in fullfilment of the requirements for the degree of Bachelor of Arts in Mathematics, Occidental College (2011)

As inputs the program requires:

  (1) A .csv (comma separated value) file of career total statistics for a team 9 players, located in the current working 
      directory, and formatted as follows:
      
      order | player_name | position | homeruns | triples | doubles | singles | walks | outs | plate_appearances 
          1 |             |          |          |         |         |         |       |      |
          2 |             |          |          |         |         |         |       |      |
          3 |             |          |          |         |         |         |       |      |
          4 |             |          |          |         |         |         |       |      |
          5 |             |          |          |         |         |         |       |      |
          6 |             |          |          |         |         |         |       |      |
          7 |             |          |          |         |         |         |       |      |
          8 |             |          |          |         |         |         |       |      |
          9 |             |          |          |         |         |         |       |      |
          
  (2) A nine-digit string of integers, 1 through 9, which reorders the 9 hitters included in the .csv file where, for example:
  
      123456789 = original order
      987654321 = reverse order
      123987456 = some other ordering
      333333333 = all the same hitter
      
And as outputs returns:

    (1) A lineup card with batting order, including player names and positions.
    
    (2) The expected numbers of runs scored per game for the given lineup.
    
    
Sources:  
Senior thesis and simulation modeled very closely off of Joel Sokol's paper "An Intuitive Markov Chain Lesson From Baseball" (Sokol, Informs Transactions on Education 2004) and accompanying MATLAB project, found here:  https://www2.isye.gatech.edu/~jsokol/markovball/

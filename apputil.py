from collections import defaultdict
import numpy as np
import random
import requests
import re


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):
        """
        Build a term dictionary of Markov states.
        
        The dictionary has:
        - Keys: unique tokens in the corpus
        - Values: list of all tokens that follow that key (including duplicates)
        
        We include duplicates because they naturally encode the transition probabilities.
        If a word follows another word multiple times, it should have a higher probability
        of being selected during generation.
        
        Returns:
            dict: The term dictionary mapping tokens to lists of following tokens
        """
        # Use defaultdict to automatically create empty lists for new keys
        self.term_dict = defaultdict(list)
        
        # Tokenize the corpus by splitting on whitespace
        tokens = self.corpus.split()
        
        # Iterate through the tokens, stopping one before the end
        # (since the last token has no following token)
        for i in range(len(tokens) - 1):
            current_token = tokens[i]
            next_token = tokens[i + 1]
            
            # Add the next token to the list of tokens that follow the current token
            # We keep duplicates to maintain proper transition probabilities
            self.term_dict[current_token].append(next_token)
        
        # Convert defaultdict back to regular dict as required by the template
        self.term_dict = dict(self.term_dict)
        
        return self.term_dict


    def generate(self, seed_term=None, term_count=15):
        """
        Generate text using the Markov property.
        
        Args:
            seed_term (str): Optional starting term. If None, a random term is chosen.
            term_count (int): Number of terms to generate (default: 15)
        
        Returns:
            str: Generated text of specified length
        
        Raises:
            ValueError: If the seed_term is not in the corpus
        """
        # Ensure term_dict exists
        if self.term_dict is None:
            self.get_term_dict()
        
        # Initialize the result list
        result = []
        
        # Handle the seed term
        if seed_term is not None:
            # Check if seed_term is in the corpus
            if seed_term not in self.term_dict:
                raise ValueError(f"Seed term '{seed_term}' not found in corpus")
            current_term = seed_term
        else:
            # Choose a random starting term from the available keys
            # We only choose from terms that have following terms
            available_starts = [term for term in self.term_dict.keys() if self.term_dict[term]]
            if not available_starts:
                return ""
            current_term = np.random.choice(available_starts)
        
        # Add the first term to the result
        result.append(current_term)
        
        # Generate the remaining terms
        for _ in range(term_count - 1):
            # Check if the current term has any following terms
            if current_term not in self.term_dict or not self.term_dict[current_term]:
                # If we hit a dead end (e.g., the last word in corpus),
                # we can either stop or pick a new random term to continue
                # Here we'll pick a new random term to continue generation
                available_terms = [term for term in self.term_dict.keys() if self.term_dict[term]]
                if not available_terms:
                    break
                current_term = np.random.choice(available_terms)
            else:
                # Choose the next term randomly from the possible following terms
                # Using numpy.random.choice for random selection
                possible_next = self.term_dict[current_term]
                current_term = np.random.choice(possible_next)
            
            result.append(current_term)
        
        # Join the terms with spaces and return
        return ' '.join(result)


def fetch_and_clean_quotes():
    """
    Fetch and clean inspirational quotes from the dataset.
    
    Returns:
        str: Cleaned corpus of quotes
    """
    # Fetch the quotes
    url = 'https://raw.githubusercontent.com/leontoddjohnson/datasets/main/text/inspiration_quotes.txt'
    
    try:
        content = requests.get(url)
        quotes_raw = content.text
    except Exception as e:
        # Fallback corpus if fetch fails
        quotes_raw = """
        "Healing comes from taking responsibility: to realize that it is you - and no one else - that creates your thoughts, your feelings, and your actions."
        "Life is a journey and if you fall in love with the journey you will be in love forever."
        "When you return to your old hometown, you find it wasn't the town you missed, but your childhood."
        "As we grow old, the beauty steals inward."
        "Life begins as a quest of the child for the man, and ends as a journey by the man to rediscover the child."
        "Ultimately your greatest teacher is to live with an open heart."
        "Doing what you like is freedom. Liking what you do is happiness."
        "We forge the chains we wear in life."
        """
    
    # Clean the quotes (following the notebook's approach)
    quotes = quotes_raw.replace('\n', ' ')
    # Split on any type of quotation mark
    quotes = re.split(r'["\'\u201c\u201d\u2018\u2019]', quotes)   # split on all quote types
    
    # Skip the first one, and capture every other element
    if len(quotes) > 1:
        quotes = quotes[1::2]
    
    # Create one long corpus of text
    corpus = ' '.join(quotes)
    
    # Remove long whitespaces
    corpus = re.sub(r"\s+", " ", corpus)
    
    # Remove leading/trailing whitespaces
    corpus = corpus.strip()
    
    return corpus
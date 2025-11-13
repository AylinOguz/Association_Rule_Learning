# Association Rule Learning (ARL) — Market Basket Analysis
![image](https://github.com/AylinOguz/Association_Rule_Learning/blob/main/arl.png?raw=true)


## Project Overview

This project demonstrates Association Rule Learning (ARL) — a popular data mining technique used to uncover hidden relationships between products that are frequently purchased together.
Using the Online Retail II dataset, the project identifies strong association rules and recommends products to users based on their current basket items.

The analysis and recommendation process are performed using the Apriori algorithm and association rules (support, confidence, and lift metrics).

## Dataset

Source: UCI Machine Learning Repository – Online Retail II

Description: This dataset contains transactional data from a UK-based online retail store between 2010 and 2011.

Main Columns:

Invoice: Unique transaction (basket) ID

StockCode: Unique product ID

Description: Product name

Quantity: Quantity of each product purchased

InvoiceDate: Transaction date

Price: Unit price of each product

Country: Customer’s country


## Project Workflow

### Data Preparation

- Load the Online Retail II dataset.

- Keep only transactions made in the United Kingdom for higher data consistency.

- Remove missing values and cancelled transactions.

- Filter out products with extremely low frequency to reduce noise.

### Data Transformation

- Convert the dataset into a basket–item matrix (each row = invoice, each column = product).

- Fill the matrix with binary values:

- 1 → the product was purchased

- 0 → the product was not purchased

### Applying the Apriori Algorithm

- Use the Apriori algorithm to identify frequent itemsets based on a defined minimum support threshold.

- These itemsets represent groups of products that are often bought together.

### Generating Association Rules

- Apply Association Rule Learning (ARL) on the frequent itemsets.

- Extract rules based on confidence and lift values to determine strong product relationships.

- Example:

  If a customer buys “Coffee,” they are also likely to buy “Mug.”

### Making Product Recommendations

- For any given product, retrieve related items using the generated association rules.

- This enables market basket analysis and cross-selling recommendations.

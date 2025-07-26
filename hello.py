import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
from crewai_tools import tool

import yfinance as yf


class Tools:

    @tool("One month stock price history")
    def stock_price(ticker):
        """
        Useful to get a month's worth of stock price data as CSV.
        The input of this tool should a ticker, for example AAPL, NET, TSLA etc...
        """
        stock = yf.Ticker(ticker)
        return stock.history(period="1mo").to_csv()

    @tool("Stock news URLs")
    def stock_news(ticker):
        """
        Useful to get URLs of news articles related to a stock.
        The input to this tool should be a ticker, for example AAPL, NET
        """
        stock = yf.Ticker(ticker)
        return list(map(lambda x: x["link"], stock.news))

    @tool("Company's income statement")
    def income_stmt(ticker):
        """
        Useful to get the income statement of a stock as CSV.
        The input to this tool should be a ticker, for example AAPL, NET
        """
        stock = yf.Ticker(ticker)
        return stock.income_stmt.to_csv()

    @tool("Balance sheet")
    def balance_sheet(ticker):
        """
        Useful to get the balance sheet of a stock as CSV.
        The input to this tool should be a ticker, for example AAPL, NET
        """
        stock = yf.Ticker(ticker)
        return stock.balance_sheet.to_csv()

    @tool("Get insider transactions")
    def insider_transactions(ticker):
        """
        Useful to get insider transactions of a stock as CSV.
        The input to this tool should be a ticker, for example AAPL, NET
        """
        stock = yf.Ticker(ticker)
        return stock.insider_transactions.to_csv()
from crewai import Agent


from crewai_tools import ScrapeWebsiteTool


class Agents:

    def technical_analyst(self):
        return Agent(
            role="Technical Analyst",
            goal="Analyses the movements of a stock and provides insights on trends, entry points, resistance and support levels.",
            backstory="An expert in technical analysis, you're known for your ability to predict stock movements and trends based on historical data. You provide valuable insights to your customers.",
            verbose=True,
            tools=[
                Tools.stock_price,
            ],
        )

    def researcher(self):
        return Agent(
            role="Researcher",
            goal="Gathers, interprets and summarizes vasts amounts of data to provide a comprehensive overview of the sentiment and news surrounding a stock.",
            backstory="You're skilled in gathering and interpreting data from various sources to give a complete picture of a stock's sentiment and news. You read each data source carefuly and extract the most important information. Your insights are crucial for making informed investment decisions.",
            verbose=True,
            tools=[
                Tools.stock_news,
                ScrapeWebsiteTool(),
            ],
        )

    def financial_analyst(self):
        return Agent(
            role="Financial Analyst",
            goal="Uses financial statements, insider trading data, and other financial metrics to evaluate a stock's financial health and performance.",
            backstory="You're a very experienced investment advisor who uses a combination of technical and fundamental analysis to provide strategic investment advice to your clients. You look at a company's financial health, market sentiment, and qualitative data to make informed recommendations.",
            verbose=True,
            tools=[
                Tools.balance_sheet,
                Tools.income_stmt,
                Tools.insider_transactions,
            ],
        )

    def hedge_fund_manager(self):
        return Agent(
            role="Hedge Fund Manager",
            goal="Manages a portfolio of stocks and makes strategic investment decisions to maximize returns using insights from financial analysts, technical analysts, and researchers.",
            backstory="You're a seasoned hedge fund manager with a proven track record of making profitable investment decisions. You're known for your ability to manage risk and maximize returns for your clients.",
            verbose=True,
        )
from crewai import Task


class Tasks:

    def research(self, agent):
        return Task(
            description="Gather and analyze the latest news and market sentiment surrounding the stock of {company}. Provide a summary of the news and any notable shifts in market sentiment.",
            expected_output=f"Your final answer MUST be a detailed summary of the news and market sentiment surrounding the stock. Include any notable shifts in market sentiment and provide insights on how these factors could impact the stock's performance.",
            agent=agent,
            output_file="stock_news.md",
        )

    def technical_analysis(self, agent):
        return Task(
            description="Conduct a detailed technical analysis of the price movements of {company}'s stock and trends identify key support and resistance levels, chart patterns, and other technical indicators that could influence the stock's future performance. Use historical price data and technical analysis tools to provide insights on potential entry points and price targets.",
            expected_output=f"Your final answer MUST be a detailed technical analysis report that includes key support and resistance levels, chart patterns, and technical indicators. Provide insights on potential entry points, price targets, and any other relevant information that could help your customer make informed investment decisions.",
            agent=agent,
            output_file="technical_analysis.md",
        )

    def finacial_analysis(self, agent):
        return Task(
            description="Analyze {company}'s financial statements, insider trading data, and other financial metrics to evaluate the stock's financial health and performance. Provide insights on the company's revenue, earnings, cash flow, and other key financial metrics. Use financial analysis tools and models to assess the stock's valuation and growth potential.",
            expected_output=f"Your final answer MUST be a detailed financial analysis report that includes insights on the company's financial health, performance, and valuation. Provide an overview of the company's revenue, earnings, cash flow, and other key financial metrics. Use financial analysis tools and models to assess the stock's valuation and growth potential.",
            agent=agent,
            output_file="financial_analysis.md",
        )

    def investment_recommendation(self, agent, context):
        return Task(
            description="Based on the research, technical analysis, and financial analysis reports, provide a detailed investment recommendation for {company}'s stock. Include your analysis of the stock's potential risks and rewards, and provide a clear rationale for your recommendation.",
            expected_output=f"Your final answer MUST be a detailed investment recommendation report to BUY or SELL the stock that includes your analysis of the stock's potential risks and rewards. Provide a clear rationale for your recommendation based on the research, technical analysis, and financial analysis reports.",
            agent=agent,
            context=context,
            output_file="investment_recommendation.md",
        )
from crewai import Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI

agents = Agents()
tasks = Tasks()

researcher = agents.researcher()
technical_analyst = agents.technical_analyst()
financial_analyst = agents.financial_analyst()
hedge_fund_manager = agents.hedge_fund_manager()

research_task = tasks.research(researcher)
technical_task = tasks.technical_analysis(technical_analyst)
financial_task = tasks.finacial_analysis(financial_analyst)
recommend_task = tasks.investment_recommendation(
    hedge_fund_manager,
    [
        research_task,
        technical_task,
        financial_task,
    ],
)

crew = Crew(
    agents=[
        researcher,
        technical_analyst,
        financial_analyst,
        hedge_fund_manager,
    ],
    tasks=[
        research_task,
        technical_task,
        financial_task,
        recommend_task,
    ],
    verbose=2,
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
    memory=True,
)

result = crew.kickoff(
    inputs=dict(
        company="Salesforce",
    ),
)

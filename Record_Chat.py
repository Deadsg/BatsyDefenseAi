import discord
from discord.ext import commands

bot = commands.Bot(command_prefix="!")

# Initialize a list to store chat history
chat_history = []

# Your other bot setup code goes here...

@bot.command()
async def record_chat(ctx, *, message):
    # Add the message to the chat history
    chat_history.append(message)
    await ctx.send(f'Message recorded: "{message}"')

# Your other bot event handling code goes here...

# Add your Discord bot token here
bot.run('YOUR_DISCORD_TOKEN_HERE')
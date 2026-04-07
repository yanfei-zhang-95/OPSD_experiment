import asyncio
from workspace.Browsecomp_zh.agent.agent import Workflow

async def main():
    print("Initializing Workflow...")
    # Using 'deepseek-chat' as configured in config/config.yaml
    agent = Workflow(
        llm_config="deepseek-chat"
    )
    
    request = "这是一首由中国知名男歌星演唱的一首流行歌曲，收录于该歌手的同名专辑中。这首歌的编曲者曾在乐团担任键盘手。该歌曲的和声歌词中出现一种食物，且该食物的最后一个字和歌名的最后一个字一样。请问这首歌曲名是什么？"
    print(f"Running agent with request: {request}")
    
    result = await agent.run(request)
    
    print("\n--- Agent Result ---")
    print(f"Answer: {result.get('answer')}")
    print("--------------------")
    
if __name__ == "__main__":
    asyncio.run(main())

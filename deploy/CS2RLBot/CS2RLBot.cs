using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Core.Attributes;

namespace CS2RLBot;

[MinimumApiVersion(80)]
public class CS2RLBotPlugin : BasePlugin
{
    public override string ModuleName    => "CS2RLBot";
    public override string ModuleVersion => "0.1.0";
    public override string ModuleAuthor  => "cs2rl";
}

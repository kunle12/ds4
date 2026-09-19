# Mac Studio and DGX Spark Network

[README](../README.md) | [DGX Spark](DGX_SPARK.md) | [Inference across machines](DISTRIBUTED.md)

A DGX Spark (`spike`) is connected to a Mac Studio (`titan`) by a direct
Ethernet cable between a built-in port on each machine. The Mac Studio acts as
the Spark's router: it forwards between the private direct link and the
192.168.0.0/24 LAN, and it translates addresses in both directions.

This document is the rebuild reference. It records the address plan, every file
the setup owns, the exact commands to reproduce it, how to verify it, and what
silently breaks it. Verified against macOS 26.7 (Mac16,9) and Ubuntu 24.04.5 LTS
(kernel 7.0.0-1019-nvidia, arm64), 2026-09-19.

The two halves are coupled. The Spark's DNS server is 192.168.0.1 on the far
side of the Mac, so the Spark resolves names **only** while the Mac's NAT rule is
loaded. Rebuilding one machine without the other leaves the Spark unable to
resolve anything, with no error message to explain it.

## Address plan

| Role | Machine | Interface | MAC | Address |
| --- | --- | --- | --- | --- |
| LAN uplink | titan | `en10` (AX88179B USB) | `9c:69:d3:ac:73:51` | `192.168.0.37/24` DHCP |
| Spark front door (VIP) | titan | `en10` alias | same | `192.168.0.38/24` static |
| Direct link, Mac side | titan | `en0` (built-in 10GbE) | `1c:1d:d3:e5:63:9d` | `192.168.2.1/24` static |
| Direct link, Spark side | spike | `enP7s7` | `fc:4c:ea:f9:71:a5` | `192.168.2.2/24` static |
| LAN router, DHCP, DNS | router | — | — | `192.168.0.1` |

The cable runs directly between the two machines with no switch in between.
`192.168.2.0/24` exists only on that cable; nothing on the LAN has a route to
it. The DHCP lease on `en10` lasts 10 days and is bound to that MAC address, as
is the rest of the LAN.

`spike.local` resolves over the direct link by mDNS, so `ssh spike.local` works
from the Mac without the LAN being involved. The Mac's LAN address is
`192.168.0.37`, and it also has Wi-Fi on `en1` at `192.168.0.82`. Both are on the
same wire, which macOS tolerates: `en10`'s LAN routes are global and `en1`'s are
interface-scoped, so forwarded traffic consistently leaves through `en10`.

## How traffic flows

Two independent translations, and both are required. Traffic in each direction
leaves by a different interface, so neither rule can substitute for the other.

| Direction | Interface in | Interface out | Rule | Why |
| --- | --- | --- | --- | --- |
| LAN to Spark | `en10` | `en0` | `rdr` | Rewrites the destination `192.168.0.38` to `192.168.2.2` |
| Spark to LAN and internet | `en0` | `en10` | `nat` | Rewrites the source `192.168.2.x` to `192.168.0.37` |

The LAN side reaches the Spark by its virtual address `192.168.0.38`. It cannot
use `192.168.2.2`, because no LAN host has a route to `192.168.2.0/24`. The
redirect is stateful, so replies from the Spark are translated back
automatically and clients never see the private address.

The Spark's own traffic needs the mirror-image rule. Forwarding alone is not
enough: without `nat`, packets leave `en10` carrying source `192.168.2.2`, which
no LAN host or router can return a reply to. The packet is delivered and the
answer goes nowhere, so DNS to `192.168.0.1` and every LAN and internet
connection fails without an error. If `rdr` is present but `nat` is missing, LAN
clients can reach the Spark's services while the Spark can reach nothing at all.

The NAT source is the real DHCP address `192.168.0.37` and deliberately not the
VIP. Pointing it at `192.168.0.38` would collide with the redirect that owns
that address.

Both directions also need `net.inet.ip.forwarding=1`. IPv6 is not forwarded
(`net.inet6.ip6.forwarding` is 0) and the Spark has no IPv6 default route, so
IPv6 is link-local only on both sides. The Spark still receives AAAA records
from DNS; clients fall back to IPv4.

## Mac Studio (titan)

### Interfaces

| Service | Device | Method | Address |
| --- | --- | --- | --- |
| Ethernet | `en0` | Manual | `192.168.2.1/24`, no router |
| Ethernet 2 | `en10` | DHCP | `192.168.0.37/24`, plus the `192.168.0.38` alias |
| Wi-Fi | `en1` | DHCP | `192.168.0.82/24` |
| Thunderbolt Bridge | `bridge0` | inactive | — |

`en0` carries no router, so all of the Mac's own traffic still leaves by `en10`.
Service names are positional after a reinstall: confirm the mapping with
`networksetup -listallhardwareports` before trusting the names above.

### Files

| Path | Mode | Owner | Purpose |
| --- | --- | --- | --- |
| `/etc/pf.conf` | 644 | `root:wheel` | NAT, redirects, filtering |
| `/etc/sysctl.conf` | 644 | `root:wheel` | `net.inet.ip.forwarding=1` |
| `/usr/local/sbin/dgx-router.sh` | 755 | `root:wheel` | Adds the VIP, loads and enables pf |
| `/Library/LaunchDaemons/com.local.pf.rules.plist` | 644 | `root:wheel` | Loads and enables pf at boot |
| `/Library/LaunchDaemons/com.qbot.dgx-router.plist` | 644 | `root:wheel` | Runs the script every 60 seconds |
| `/etc/pf.conf.pre-nat` | 644 | `root:wheel` | Backup of the ruleset before NAT was added |
| `/var/log/dgx-router.log` | 644 | `root:wheel` | Output of the 60-second loop |

`/etc/pf.conf`:

```text
#
# pf.conf -- Mac Studio as gateway for the DGX Spark ("spike")
#
#   en10 : LAN side   -- 192.168.0.0/24 (DHCP 192.168.0.37) + Spark VIP 192.168.0.38
#   en0  : Spark side -- 192.168.2.0/24 direct 10GbE link (Mac .1 <-> Spark .2)
#
# LAN  -> Spark : arrives on $WAN_IF addressed to the VIP, redirected (rdr) to the Spark.
# Spark -> LAN  : arrives on $DGX_IF, source 192.168.2.x is unroutable on the LAN, so it
#                 must be rewritten to the WAN address (nat) on the way out $WAN_IF.
#                 Without this the LAN hosts cannot route replies back to the Spark
#                 (DNS to 192.168.0.1, or any 192.168.0.x host, will silently fail).
#
# Companion: /usr/local/sbin/dgx-router.sh adds $DGX_VIP, loads this file, enables pf.
# Requires net.inet.ip.forwarding=1 (/etc/sysctl.conf).

# ---- Macros ----------------------------------------------------------------
WAN_IF   = "en10"             # LAN side
DGX_IF   = "en0"              # direct link to the Spark

DGX_NET  = "192.168.2.0/24"   # Spark side network
DGX_REAL = "192.168.2.2"      # Spark's address on the direct link
DGX_VIP  = "192.168.0.38"     # LAN-side address that fronts the Spark
LAN_NET  = "192.168.0.0/24"   # LAN side network

# Source address used when the Spark talks to the LAN. Must be the real DHCP
# address of $WAN_IF -- NOT $DGX_VIP, which is reserved as the Spark's front door.
WAN_ADDR = "192.168.0.37"

# ---- Options ---------------------------------------------------------------
set skip on lo0

# ---- com.apple anchor point ------------------------------------------------
scrub-anchor "com.apple/*"

# ---- NAT -------------------------------------------------------------------
nat-anchor "com.apple/*"

# Spark -> LAN/internet: rewrite source 192.168.2.x to the WAN address.
nat on $WAN_IF inet from $DGX_NET to any -> $WAN_ADDR

# ---- Redirects -------------------------------------------------------------
rdr-anchor "com.apple/*"

# LAN -> Spark: the Spark has no LAN-routable address of its own, so anything
# addressed to the VIP is forwarded to its real address on the direct link.
rdr on $WAN_IF inet proto tcp  from any to $DGX_VIP -> $DGX_REAL
rdr on $WAN_IF inet proto udp  from any to $DGX_VIP -> $DGX_REAL
rdr on $WAN_IF inet proto icmp from any to $DGX_VIP -> $DGX_REAL

# ---- Dummynet --------------------------------------------------------------
dummynet-anchor "com.apple/*"

# ---- Filtering -------------------------------------------------------------
# LAN side: clients on the LAN reach this machine and the forwarded Spark traffic.
pass in  quick on $WAN_IF inet from $LAN_NET
pass out quick on $WAN_IF inet

# Spark side: the Spark reaches this machine and, via NAT, the LAN down $WAN_IF.
pass in  quick on $DGX_IF inet from $DGX_NET
pass out quick on $DGX_IF inet

anchor "com.apple/*"
load anchor "com.apple" from "/etc/pf.anchors/com.apple"
```

`/usr/local/sbin/dgx-router.sh`:

```sh
#!/bin/sh

WAN=en10
DGX_VIP=192.168.0.38
MASK=255.255.255.0

# Wait for the interface to be available
for i in $(seq 1 60); do
    if /sbin/ifconfig "$WAN" | grep -q "status"; then
        break
    fi
    sleep 1
done

# Add the alias IP if it is not already present
if ! /sbin/ifconfig "$WAN" | grep -q "$DGX_VIP"; then
    /sbin/ifconfig "$WAN" inet "$DGX_VIP" "$MASK" alias
fi

# Load and enable pf rules
/sbin/pfctl -f /etc/pf.conf
/sbin/pfctl -e 2>/dev/null || true

exit 0
```

`/Library/LaunchDaemons/com.local.pf.rules.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.local.pf.rules</string>
    <key>ProgramArguments</key>
    <array>
        <string>/sbin/pfctl</string>
        <string>-ef</string>
        <string>/etc/pf.conf</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
```

`/Library/LaunchDaemons/com.qbot.dgx-router.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.qbot.dgx-router</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/sbin/dgx-router.sh</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>StartInterval</key>
    <integer>60</integer>

    <key>StandardOutPath</key>
    <string>/var/log/dgx-router.log</string>

    <key>StandardErrorPath</key>
    <string>/var/log/dgx-router.log</string>
</dict>
</plist>
```

### Load order at boot

Three daemons touch the ruleset. They are redundant on purpose.

| Daemon | Origin | Action | Enables pf |
| --- | --- | --- | --- |
| `com.apple.pfctl` | stock macOS | `pfctl -f /etc/pf.conf` at boot | no |
| `com.local.pf.rules` | this setup | `pfctl -ef /etc/pf.conf` at boot | yes |
| `com.qbot.dgx-router` | this setup | alias, `pfctl -f`, `pfctl -e` every 60 seconds | yes |

The 60-second loop is the safety net: it re-adds the VIP, reloads the ruleset
and re-enables pf, so the setup recovers within a minute from a flush, a
disable, or a lost alias, without a reboot. Because it reloads `/etc/pf.conf`
unconditionally, editing that file is enough to change the behaviour; nothing
has to be restarted.

### Rebuild

```sh
# 1. Direct link to the Spark. Static, no router.
networksetup -setmanual "Ethernet" 192.168.2.1 255.255.255.0

# 2. LAN uplink keeps DHCP, which supplies 192.168.0.37 and the 192.168.0.1 router.
networksetup -setdhcp "Ethernet 2"

# 3. Forwarding between the two interfaces.
printf 'net.inet.ip.forwarding=1\n' | sudo tee /etc/sysctl.conf
sudo sysctl -w net.inet.ip.forwarding=1

# 4. Ruleset, helper script, and daemons. Recreate the five files above first.
sudo install -m 644 -o root -g wheel pf.conf   /etc/pf.conf
sudo install -m 755 -o root -g wheel dgx-router.sh /usr/local/sbin/dgx-router.sh
sudo install -m 644 -o root -g wheel com.local.pf.rules.plist  /Library/LaunchDaemons/
sudo install -m 644 -o root -g wheel com.qbot.dgx-router.plist /Library/LaunchDaemons/

# 5. Register them, then apply immediately.
sudo launchctl enable system/com.local.pf.rules
sudo launchctl enable system/com.qbot.dgx-router
sudo launchctl bootstrap system /Library/LaunchDaemons/com.local.pf.rules.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.qbot.dgx-router.plist
sudo pfctl -ef /etc/pf.conf
```

### Verification

```sh
sudo pfctl -s info | head -1          # Status: Enabled
sudo pfctl -sn                        # the nat rule
sudo pfctl -sr                        # the redirect and filter rules
ifconfig en10 | grep 'inet '          # both 192.168.0.37 and 192.168.0.38

# Proof the NAT rule is live, without any credentials and without the Spark.
# 192.168.2.1 is unroutable on the LAN, so replies are only possible if the
# source is being rewritten. Expect 0% loss; 100% loss means NAT is not loaded.
ping -c 3 -S 192.168.2.1 192.168.0.1

launchctl print system/com.local.pf.rules  | grep -E 'runs =|last exit code'
launchctl print system/com.qbot.dgx-router | grep -E 'runs =|last exit code'
```

`com.local.pf.rules` reports `runs = 1` on a machine that has just booted, and
the 60-second daemon's `runs` climbs by one per minute. Both should show
`last exit code = 0`.

## DGX Spark (spike)

### Interface

`enP7s7` is the port on the direct link. It is configured statically, with no
DHCP anywhere in the path:

| Setting | Value |
| --- | --- |
| Address | `192.168.2.2/24` |
| Gateway | `192.168.2.1` (the Mac) |
| DNS | `192.168.0.1` (the LAN router, reached through the Mac's NAT) |

`enP7s7` reports the same MAC address as `enp1s0f0np0`, one of the ConnectX
ports. Configure only one of the two; configuring both duplicates the address.

DNS was the part that took longest to find, because a resolver with no upstream
fails without an error. `/etc/resolv.conf` is a symlink into systemd-resolved
and points at the stub on `127.0.0.53`, so every ordinary lookup goes through
that stub. If the link has no DNS server, the stub has nowhere to forward and
returns nothing at all — a lookup against the router directly still succeeds, so
the network looks healthy while nothing resolves.

### NetworkManager profile

| Property | Value |
| --- | --- |
| Connection | `Wired connection 3` |
| UUID | `51c9fbce-209f-3fcd-beb2-159da793af97` |
| Interface | `enP7s7` |
| `ipv4.method` | `manual` |
| `ipv4.addresses` | `192.168.2.2/24` |
| `ipv4.gateway` | `192.168.2.1` |
| `ipv4.dns` | `192.168.0.1` |

The profile is stored by NetworkManager as netplan YAML at
`/etc/netplan/90-NM-51c9fbce-209f-3fcd-beb2-159da793af97.yaml`, mode 600, owned
by root. It is not in `/etc/NetworkManager/system-connections`, which is empty:

```
/etc/netplan/90-NM-<uuid>.yaml          persistent profile
        |  netplan ships as a systemd generator, so it runs on every boot
        |  before any unit starts, and renders the profile for the backend
        v
/run/NetworkManager/system-connections/netplan-NM-<uuid>.nmconnection
        |  NetworkManager reads it via plugins=ifupdown,keyfile
        v
systemd-resolved (enabled at boot) publishes 192.168.0.1 on link enP7s7
```

Because `/run` is cleared on every boot and repopulated by that generator from
the netplan file, the netplan file is the only thing that must survive. Editing
it by hand is unnecessary: `nmcli` writes it, and the generator renders it.

`/etc/netplan/00-installer-config.yaml` is the installer's baseline. It is mode
600 and cannot be read without root, so its contents are not recorded here. The
port settings that matter live in the NetworkManager profile above: `nmcli`
rewrote that file when the address and DNS were set, and it is the file the
generator renders from. Keeping the address, gateway and DNS out of the
installer's baseline is what stops `netplan apply` from reverting them.

### Rebuild

```sh
# 1. Static address, gateway and DNS on the direct link.
sudo nmcli con add type ethernet con-name spark-direct ifname enP7s7 \
  ipv4.method manual \
  ipv4.addresses 192.168.2.2/24 \
  ipv4.gateway 192.168.2.1 \
  ipv4.dns 192.168.0.1
sudo nmcli con up spark-direct

# 2. Remove any auto-created profile for the same port, so two profiles
#    cannot fight over it.
nmcli -t -f NAME,DEVICE con show | grep enP7s7
# sudo nmcli con delete "<the other name>"

# 3. Remote access. Ubuntu 24.04 activates sshd by socket, so ssh.service
#    reports disabled while ssh.socket is enabled. That is normal.
sudo systemctl enable --now ssh.socket

# 4. Passwordless SSH from the Mac: append the Mac's public key.
#    The Mac's key is ~/.ssh/id_ed25519.pub.
cat >> ~/.ssh/authorized_keys <<'EOF'
<paste the contents of ~/.ssh/id_ed25519.pub from titan>
EOF
```

The connection name is cosmetic; the UUID changes when the profile is
recreated, and the netplan filename follows it. Nothing else refers to either.

### Verification

```sh
ip route get 192.168.0.1                 # via 192.168.2.1 dev enP7s7 src 192.168.2.2
resolvectl dns enP7s7                    # 192.168.0.1
resolvectl status | sed -n '/enP7s7/,+4p'
#   Current Scopes: DNS
#   Protocols: +DefaultRoute
#   DNS Servers: 192.168.0.1
getent ahostsv4 google.com               # an address, exit status 0
ping -c 2 192.168.0.1                    # the LAN, no DNS involved
ping -4 -c 2 google.com                  # name resolution and reachability together
```

`resolvectl status` should show `+DefaultRoute`. If it shows `-DefaultRoute`
the link's DNS is not being used for general lookups, and the missing piece is
`ipv4.dns-search ~.`. NetworkManager normally sets that on its own, because this
profile holds the default route.

## Diagnosing a broken path

Work from the Spark outwards; each command narrows the fault by one hop.

| Observation | Fault | Where to look |
| --- | --- | --- |
| `ping 192.168.2.1` fails | The cable, a port, or the Mac's `en0` address | Both interfaces, link lights |
| `ping 192.168.2.1` works, `ping 192.168.0.1` fails | Missing or unloaded `nat`, or forwarding off | `sudo pfctl -sn`, `sysctl net.inet.ip.forwarding`, `/var/log/dgx-router.log` |
| `ping 192.168.0.1` works, `dig @192.168.0.1` fails | The router's resolver, not this setup | The router |
| `dig @192.168.0.1` works, `getent hosts` returns nothing | The Spark's link has no DNS server | `resolvectl status`, `nmcli con show "Wired connection 3"` |
| `getent` works, `ping google.com` fails | Upstream, not this setup | The router's own internet access |
| The Spark reaches nothing, LAN clients reach the Spark fine | Exactly the missing-`nat` symptom | `sudo pfctl -sn` for the `nat` line |
| LAN clients cannot reach `192.168.0.38` | The `rdr` rules, the VIP alias, or PF disabled | `sudo pfctl -s rdr`, `sudo pfctl -s info`, `ifconfig en10` |
| Everything worked, then stopped after an update | `/etc/pf.conf` replaced | `grep -n 'nat on' /etc/pf.conf`, restore from `/etc/pf.conf.pre-nat` |

`ping -S 192.168.2.1 192.168.0.1` on the Mac tests the NAT rule alone, with no
dependency on the Spark and no credentials. It is the fastest way to tell
whether the Mac half is intact.

## Known limits

- **The NAT source is pinned.** `WAN_ADDR` in `/etc/pf.conf` is the literal
  `192.168.0.37`. Every LAN address here is bound to a MAC, so the lease should
  not move, but if it ever does, NAT keeps rewriting to a stale address and the
  Spark breaks exactly as it did before the rule existed. Update the macro, or
  derive it in `dgx-router.sh`, which already runs every 60 seconds.
- **The Spark depends on the Mac.** The Mac is its router and its route to DNS.
  Boot the Spark with the Mac down and it comes up with no DNS and no LAN or
  internet access until the Mac returns. Nothing on the Spark can fix that; it
  is the topology.
- **Interface names can move.** `en10` is a USB adapter and `en0` numbering is
  assigned by hardware path. After a reinstall, confirm the mapping and update
  `WAN_IF` and `DGX_IF` in `/etc/pf.conf` together with `WAN` in
  `dgx-router.sh`.
- **`192.168.0.38` must stay reserved.** If the DHCP server ever hands that
  address to another device, that device and the Spark both answer for it. Keep
  it out of the pool, or out of any range the router hands out.
- **macOS updates may replace `/etc/pf.conf`.** It is the one file here that the
  system considers its own. After a major upgrade, check for the `nat` line.
- **DNS resolution is only as good as the far side.** The Spark's resolver
  points at `192.168.0.1`, and reaching it depends on this setup. If the router's
  resolver is ever unavailable, the Spark resolves nothing, and this setup is not
  at fault.

## Reverting

The original ruleset, before NAT was added, is kept at `/etc/pf.conf.pre-nat`:

```sh
sudo install -m 644 -o root -g wheel /etc/pf.conf.pre-nat /etc/pf.conf
sudo pfctl -f /etc/pf.conf
```

That removes the `nat` rule and leaves the Spark able to receive connections on
the VIP but unable to resolve anything or reach the LAN, which is the state
before this was fixed.

To undo the Spark side:

```sh
sudo nmcli con mod "Wired connection 3" ipv4.dns ""
sudo nmcli dev reapply enP7s7
```

To remove the setup from the Mac entirely:

```sh
sudo launchctl bootout system/com.qbot.dgx-router
sudo launchctl bootout system/com.local.pf.rules
sudo rm /Library/LaunchDaemons/com.qbot.dgx-router.plist \
        /Library/LaunchDaemons/com.local.pf.rules.plist \
        /usr/local/sbin/dgx-router.sh
sudo pfctl -d
sudo ifconfig en10 inet 192.168.0.38 delete
```

Removing `net.inet.ip.forwarding=1` from `/etc/sysctl.conf` as well stops the Mac
forwarding between its own interfaces.

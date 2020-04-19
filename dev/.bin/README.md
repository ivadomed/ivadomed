
To add more dev scripts, use a symlink:

ln -s ../your-script.ext your-script

Make sure you've added a proper shebang line to ../your-script.ext
(e.g. #!/usr/bin/env python, or #!/bin/sh or #!/usr/bin/env bash, depending on your language)

and that you've done

chmod +x ../your-script.ext

You could also put the scripts directly in here, but separating them out with a symlink is friendlier
to IDEs that use the file extension to determine what settings to use.

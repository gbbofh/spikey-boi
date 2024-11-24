#!/usr/bin/env python3


import spikeyboi.app


def main():
    app = spikeyboi.app.App('Spikey-Boi', (1024, 768))
    app.main_loop()

if __name__ == '__main__':
    main()

while True:
    try:
        x = int(input('Enter a number:'))
        print(x)
        break
    except ValueError:
        print('That\'s not a valid number')
    except KeyboardInterrupt:
        print('Not input taken')
        break
    finally:
        print('\nAttempted Input\n')
# encoding: utf-8
'''
○×ゲームのマーク
'''
class MarkBase:
    def is_empty(self):
        return True

    def opponent(self):
        return 0

    def to_int(self):
        return 0

    def to_string(self):
        return '.'

if __name__ == '__main__':
    mark = MarkBase()
    print mark.is_empty()
    print mark.opponent()
    print mark.to_int()
    print mark.to_string()

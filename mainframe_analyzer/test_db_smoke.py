from modules.database_manager import DatabaseManager


def run_test():
    db = DatabaseManager('c:\\Users\\Admin\\analyzer\\mainframe_analyzer\\data\\mainframe_analyzer.db')
    db.initialize_database()
    session = 'test-session-1'
    db.create_session(session, 'test-project')
    layout = {
        'name': 'TEST-LAYOUT',
        'level': '01',
        'line_start': 1,
        'line_end': 10,
    'source_code': '''01 TEST-LAYOUT.
       05 FIELD-ONE.
       05 FIELD-TWO.''',
        'fields': [
            {'name': 'FIELD-ONE', 'operation_type': 'input', 'line_number': 2},
            {'name': 'FIELD-TWO', 'operation_type': 'output', 'line_number': 3}
        ]
    }
    db.store_record_layout(session, layout, 'TEST-PROG')
    print('Stored record layout and fields without error')

if __name__ == '__main__':
    run_test()

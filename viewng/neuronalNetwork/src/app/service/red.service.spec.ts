import { TestBed } from '@angular/core/testing';

import { RedService } from './red.service';

describe('RedService', () => {
  let service: RedService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(RedService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__CSVReader__h__
#define __PUJ_ML__CSVReader__h__

#include <string>
#include <vector>

class CSVReader
{
public:
  using Self = CSVReader;

public:
  CSVReader( const std::string& fname, const std::string& delimiters = "," );
  virtual ~CSVReader( ) = default;

  void read( );

  template< class _TMatrixX, class _TMatrixY >
  void cast( _TMatrixX& X, _TMatrixY& Y, const int& p ) const;

protected:
  std::string m_FileName;
  std::string m_Delimiters;
  std::vector< std::vector< std::string > > m_Data;
};

#endif // __PUJ_ML__CSVReader__h__

// eof - $RCSfile$

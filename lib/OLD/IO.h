// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivq__eigen__IO__h__
#define __ivq__eigen__IO__h__

namespace ivq
{
  namespace eigen
  {
    namespace IO
    {
      /// Read a CSV file into a Eigen::Matrix
      template< class _TMatrix >
      void readCSV(
        _TMatrix& X,
        const std::string& filename,
        int ignore_first_rows = -1
        );
    } // end namespace
  } // end namespace
} // end namespace

#endif // __ivq__eigen__IO__h__

// eof - $RCSfile$
